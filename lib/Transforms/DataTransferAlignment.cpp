//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// DataTransferAlignment: Align ktdf.data_transfer ops to hardware requirements.
//
// Reads word size and access alignment constraints from the ktdf_arch.device
// spec and rewrites any transfers that violate those constraints, resizing
// staging buffers and transfer shapes until all transfers are legal.
//
// Ordering: After stage-coarsening + canonicalize. Before double-buffering.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <numeric>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Identifies illegal ktdf.data_transfer ops and builds a per-pipeline
/// analysis tree used by the corrective rewrite phase.
class DataTransferLegality {
 public:
  struct PipelineAnalysis;  // forward declaration, mutually recursive with
                            // StageAnalysis

  /// Describes a single ktdf.data_transfer op. Flags are set at analysis time
  /// so the rewrite phase needs no re-analysis.
  struct TransferStep {
    mlir::ktdf::DataTransferOp transfer;  // the ktdf.data_transfer op
    bool is_illegal = false;  // source memref has non-contiguous stride
    bool is_displaced =
        false;  // peer of an illegal transfer in the same pipeline
    bool needs_splat = false;  // local->FIFO: splat scalar to vector
  };

  /// Per-stage node. Exactly one of nested_pipeline or transfers is populated.
  struct StageAnalysis {
    mlir::ktdf::StageOp stage;
    std::unique_ptr<PipelineAnalysis>
        nested_pipeline;  // non-null if stage wraps a nested pipeline
    llvm::SmallVector<TransferStep>
        transfers;  // empty when nested_pipeline is set
    mlir::scf::ForOp
        innermost_loop;  // innermost scf.for enclosing the transfers
  };

  /// Whether the pass needs to widen transfers to cover a full contiguous
  /// block, or shrink them to a single element (element-addressable memory).
  enum class AlignmentKind { Widen, Shrink };

  /// Root node of a per-pipeline analysis tree. Shared parameters
  /// (alignment_kind, align_dim, alignment_factor) are stored here so all child
  /// nodes can read them directly.
  struct PipelineAnalysis {
    mlir::ktdf::PipelineOp pipeline;
    llvm::SmallVector<mlir::memref::AllocOp>
        allocs;  // staging buffer allocs in ktdf.private
    AlignmentKind alignment_kind = AlignmentKind::Widen;
    int64_t align_dim = -1;  // which dim to align, offset from end
    int64_t alignment_factor =
        0;  // factor to multiply the data transfer size by on align_dim
    int64_t stride_elems = 0;  // full contiguous block extent in elements
    llvm::SmallVector<StageAnalysis> stages;  // one entry per ktdf.stage
  };

  /// Build a PipelineAnalysis for `pipeline`. Stages containing a nested
  /// ktdf.pipeline are recorded as stubs (PipelineOp handle only); fixPipeline
  /// calls analyzePipeline on them after applying outer-level corrections.
  ///
  /// When `inherited_strides` is non-empty the legality check uses those stride
  /// values instead of reading them from the actual memref layout. This lets
  /// the outer pipeline's non-contiguous stride be visible to nested-pipeline
  /// analysis without touching any IR types.
  PipelineAnalysis analyzePipeline(
      mlir::ktdf::PipelineOp pipeline,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    PipelineAnalysis pa{};
    pa.pipeline = pipeline;

    // Collect staging buffer allocs from the ktdf.private region.
    if (auto private_op = pipeline.getPrivateOp()) {
      private_op->walk(
          [&](mlir::memref::AllocOp alloc) { pa.allocs.push_back(alloc); });
    }

    // Visit every stage.
    bool has_illegal = false;
    for (auto stage : pipeline.getStages()) {
      pa.stages.push_back(
          analyzeStage(stage, resource_kinds, has_illegal, inherited_strides));
    }

    if (!has_illegal) return pa;

    // Mark every non-illegal transfer as displaced.
    for (auto& sa : pa.stages) {
      for (auto& ts : sa.transfers) {
        if (!ts.is_illegal) ts.is_displaced = true;
      }
    }

    // Derive align_dim, alignment_factor, and alignment_kind from the first
    // illegal transfer.
    //
    // The required transfer size is determined by the non-contiguous stride on
    // the source memref: to move the data the transfer requests, we must pull
    // a contiguous block of (innermostStride) elements from the source memory.
    // The arch spec granularity is then a validity check that this
    // stride-derived size is actually a legal transfer size for the hardware —
    // it is not the source of truth for the size.
    for (auto& sa : pa.stages) {
      for (auto& ts : sa.transfers) {
        if (!ts.is_illegal) continue;

        // Select the strided operand: prefer source if it is a non-unit-stride
        // memref; fall back to destination (store-back to strided global
        // memory). Only switch to destination when it is actually a memref.
        mlir::ktdf::DataTransferOp dt = ts.transfer;
        bool use_dest = false;
        if (dt.isSourceMemRef()) {
          auto srcMrt = mlir::cast<mlir::MemRefType>(dt.getSource().getType());
          auto stride = innermostStride(srcMrt);
          if (!mlir::failed(stride) && *stride == 1 && dt.isDestMemRef())
            use_dest = true;  // source is contiguous — illegality is on dest
        } else if (dt.isDestMemRef()) {
          use_dest = true;  // source is a FIFO, destination carries the stride
        } else {
          continue;  // neither operand is a memref; nothing to derive here
        }

        auto strided_memref =
            use_dest
                ? mlir::cast<mlir::MemRefType>(dt.getDestination().getType())
                : mlir::cast<mlir::MemRefType>(dt.getSource().getType());
        auto space = strided_memref.getMemorySpace();
        auto elem_bytes =
            scheduler::tryGetSizeInBytes(strided_memref.getElementType());
        if (!elem_bytes || *elem_bytes == 0) break;

        auto granularities =
            getAccessGranularity(ts.transfer, space, resource_kinds);
        bool is_single_element = false;
        if (granularities) {
          auto word_bytes = getWordSize(ts.transfer, space, resource_kinds);
          if (word_bytes) {
            for (auto entry : *granularities) {
              uint64_t gran_bytes = entry.getSizeInWords() * *word_bytes;
              if (gran_bytes == *elem_bytes) {
                is_single_element = true;
                break;
              }
            }
          }
        } else if (!inherited_strides.empty()) {
          is_single_element = true;
        }

        if (is_single_element) {
          // Shrink: element-addressable memory, collapse to one element.
          // align_dim=0 means the innermost dimension (offset 0 from end).
          pa.alignment_kind = AlignmentKind::Shrink;
          pa.align_dim = 0;
          pa.alignment_factor = 1;
        } else {
          // Widen: the non-contiguous innermost stride tells us the total span
          // of the strided operand in elements. Each transfer iteration already
          // covers elems_per_iter (the innermost transfer size), so the number
          // of loop iterations — and hence the loop bound E — is stride /
          // elems_per_iter. Use inherited_strides when set (nested pipeline
          // case), otherwise read the stride directly from the strided memref
          // layout.
          int64_t stride_elems = 1;
          if (!inherited_strides.empty()) {
            stride_elems = inherited_strides.back();
          } else {
            auto stride = innermostStride(strided_memref);
            if (!mlir::failed(stride)) stride_elems = *stride;
          }

          // Elements covered per loop iteration = the transfer size at
          // align_dim (offset 1 from the end, i.e. sizes[rank-2]). E =
          // stride_elems / elems_per_iter gives the number of iterations needed
          // to cover one full contiguous block.
          int64_t elems_per_iter = 1;
          auto sizes = use_dest ? dt.getStaticDestSizesArray()
                                : dt.getStaticSourceSizesArray();
          if (sizes) {
            // align_dim=1 → index from end = 1 → absolute index = rank-2
            int64_t rank = (int64_t)sizes->size();
            int64_t idx = rank - 1 - 1;  // rank - 1 - align_dim(=1)
            if (idx >= 0 && idx < rank)
              elems_per_iter = std::max<int64_t>(1, (*sizes)[idx]);
          }

          // The widened dimension is the strided operand's rows: its
          // second-to-last dim, when that dim is the contiguous one (strides
          // [.., 1, S]). Widening covers that dim's whole extent R. For a
          // contiguous column-major block S == R, so the innermost stride
          // gives the same span; with columns further apart (S > R, e.g. a
          // rank-3 device layout that puts another dim between them) only R
          // is right. stride_elems stays the innermost stride: it is what the
          // nested pipeline inherits to classify its own transfers.
          int64_t span = stride_elems;
          if (inherited_strides.empty()) {
            llvm::SmallVector<int64_t> view_strides;
            int64_t view_offset = 0;
            const int64_t view_rank = strided_memref.getRank();
            if (view_rank >= 2 &&
                mlir::succeeded(strided_memref.getStridesAndOffset(
                    view_strides, view_offset)) &&
                view_strides[view_rank - 2] == 1 &&
                !mlir::ShapedType::isDynamic(
                    strided_memref.getShape()[view_rank - 2]))
              span = strided_memref.getShape()[view_rank - 2];
          }

          // align_dim=1 means the second-to-last dimension (offset 1 from end).
          pa.alignment_kind = AlignmentKind::Widen;
          pa.align_dim = 1;
          pa.stride_elems = stride_elems;
          pa.alignment_factor = span / elems_per_iter;
        }
        break;
      }
      if (pa.align_dim != -1) break;
    }

    // Post-analysis fixup: set needs_splat on memref->FIFO transfers now that
    // alignment_factor is known. Only units with ktdf_arch.feature.simd { splat
    // } can legally broadcast a scalar element across a FIFO slot. If the unit
    // lacks that capability the transfer is unresolvable — emit an error.
    // classifyTransferStep returns early for FIFO sources so this flag must
    // be applied here.
    if (pa.alignment_factor > 0) {
      for (auto& sa : pa.stages) {
        for (auto& ts : sa.transfers) {
          mlir::ktdf::DataTransferOp dt = ts.transfer;
          if (dt.isSourceMemRef() && dt.isDestFifo()) {
            if (canSplat(dt, resource_kinds)) {
              ts.needs_splat = true;
            } else {
              dt->emitError(
                  PASS_NAME
                  ": memref→FIFO transfer requires splat but the enclosing "
                  "unit does not declare ktdf_arch.feature.simd { splat }");
            }
          }
        }
      }
    }

    return pa;
  }

 private:
  /// Returns the single applicable unit kind for the stage enclosing `op`,
  /// stopping at any PipelineOp boundary. Returns nullptr if the stage has
  /// zero or more than one unit.
  mlir::ktdf_arch::KindAttr getUnitKind(mlir::Operation* op) {
    mlir::Operation* cursor = op->getParentOp();
    while (cursor && !mlir::isa<mlir::ktdf::StageOp>(cursor)) {
      if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) return nullptr;
      cursor = cursor->getParentOp();
    }
    auto stage = mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(cursor);
    if (!stage) return nullptr;
    auto units = stage.getApplicableUnits();
    if (!units || units->size() != 1) return nullptr;
    return mlir::dyn_cast<mlir::ktdf_arch::KindAttr>((*units)[0]);
  }

  /// Returns the Load feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Load> getLoad(
      mlir::Operation* op,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat = resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns the Store feature for the unit enclosing `op`.
  std::optional<mlir::ktdf_arch::feature::Store> getStore(
      mlir::Operation* op,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return std::nullopt;
    auto feat =
        resource_kinds.getFeature<mlir::ktdf_arch::feature::Store>(kind);
    if (!feat) return std::nullopt;
    return feat;
  }

  /// Returns true if the unit enclosing `op` has a SIMD feature with splat
  /// capability declared in the arch spec.
  bool canSplat(mlir::Operation* op,
                const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    auto kind = getUnitKind(op);
    if (!kind) return false;
    auto simd = resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(kind);
    return simd && simd.canSplat();
  }

  /// Returns the word size in bytes for the load unit accessing `space`.
  uint64_t getWordSize(mlir::ktdf_arch::feature::Load load,
                       mlir::Attribute space) {
    return load.getWordSize(space);
  }

  /// Returns the word size in bytes for the store unit accessing `space`.
  uint64_t getWordSize(mlir::ktdf_arch::feature::Store store,
                       mlir::Attribute space) {
    return store.getWordSize(space);
  }

  /// Returns the access granularity list for `load` and `space`.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(mlir::ktdf_arch::feature::Load load,
                       mlir::Attribute space) {
    auto list = load.getAccessGranularity(space);
    if (!list) return std::nullopt;
    return list;
  }

  /// Returns the access granularity list for `store` and `space`.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(mlir::ktdf_arch::feature::Store store,
                       mlir::Attribute space) {
    auto list = store.getAccessGranularity(space);
    if (!list) return std::nullopt;
    return list;
  }

  /// Returns the word size in bytes for `op`'s unit accessing `space`.
  /// Returns nullopt only when the unit has no Load or Store feature at all.
  /// Tries load feature first, then store.
  std::optional<uint64_t> getWordSize(
      mlir::ktdf::DataTransferOp op, mlir::Attribute space,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    if (auto load = getLoad(op, resource_kinds))
      return getWordSize(*load, space);
    if (auto store = getStore(op, resource_kinds))
      return getWordSize(*store, space);
    return std::nullopt;
  }

  /// Returns the access granularity list for `op`'s unit accessing `space`.
  /// Tries load feature first, then store.
  std::optional<mlir::ktdf_arch::AccessGranularityListAttr>
  getAccessGranularity(mlir::ktdf::DataTransferOp op, mlir::Attribute space,
                       const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
    if (auto load = getLoad(op, resource_kinds))
      return getAccessGranularity(*load, space);
    if (auto store = getStore(op, resource_kinds))
      return getAccessGranularity(*store, space);
    return std::nullopt;
  }

  /// Returns the innermost stride of `memref`. Returns 1 for identity/default
  /// row-major layouts. Returns failure() if the innermost stride is dynamic.
  llvm::FailureOr<int64_t> innermostStride(mlir::MemRefType memref) {
    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (mlir::failed(memref.getStridesAndOffset(strides, offset)) ||
        strides.empty())
      return 1;  // no explicit layout, implicit row-major
    int64_t s = strides.back();
    if (mlir::ShapedType::isDynamic(s)) return mlir::failure();
    return s;
  }

  /// Checks whether a memref operand of a data_transfer is legal against the
  /// arch spec. Checks contiguity and granularity.
  ///
  /// When `inherited_strides` is non-empty those values are used in place of
  /// the memref's actual layout strides for the contiguity check.
  ///
  /// Returns true if legal, false if illegal but correctable, nullopt on
  /// hard failure (error already emitted).
  std::optional<bool> checkMemRef(
      mlir::ktdf::DataTransferOp dt, mlir::MemRefType memref,
      llvm::ArrayRef<int64_t> transfer_sizes,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    auto space = memref.getMemorySpace();

    auto word_bytes = getWordSize(dt, space, resource_kinds);
    auto granularities = getAccessGranularity(dt, space, resource_kinds);

    // Element byte size must be statically known.
    auto elem_bytes = scheduler::tryGetSizeInBytes(memref.getElementType());
    if (!elem_bytes) {
      dt->emitError(PASS_NAME ": element type has unknown size: ")
          << memref.getElementType();
      return std::nullopt;
    }

    // Every granularity entry must cover a whole number of elements:
    // entry.sizeInWords * word_bytes must be divisible by elem_bytes.
    if (granularities) {
      for (auto entry : *granularities) {
        uint64_t gran_bytes = entry.getSizeInWords() * *word_bytes;
        if (gran_bytes % *elem_bytes != 0) {
          dt->emitError(PASS_NAME ": granularity entry size (")
              << gran_bytes << "B) is not a multiple of element size ("
              << *elem_bytes << "B) for memory space " << space;
          return std::nullopt;
        }
      }
    }

    // Use inherited strides (from an outer pipeline) when provided; otherwise
    // read the innermost stride from the memref layout.
    int64_t stride_val = 1;
    if (!inherited_strides.empty()) {
      stride_val = inherited_strides.back();
    } else {
      auto stride = innermostStride(memref);
      if (mlir::failed(stride)) {
        dt->emitError(PASS_NAME ": memref has a dynamic innermost stride");
        return std::nullopt;
      }
      stride_val = *stride;
    }

    // Total bytes being transferred (product of all sizes × element size).
    int64_t total_elems = 1;
    for (int64_t s : transfer_sizes) total_elems *= s;
    uint64_t transfer_bytes = static_cast<uint64_t>(total_elems) * *elem_bytes;

    // Find a granularity entry that the transfer size is a multiple of.
    // A transfer of N bytes is legal if N is a positive integer multiple of
    // some entry's granularity.
    if (granularities) {
      bool matched = false;
      uint64_t matched_gran_bytes = 0;
      for (auto entry : *granularities) {
        uint64_t gran_bytes = entry.getSizeInWords() * *word_bytes;
        if (gran_bytes != 0 && transfer_bytes % gran_bytes == 0) {
          matched = true;
          matched_gran_bytes = gran_bytes;
          break;
        }
      }
      if (!matched) {
        dt->emitError(PASS_NAME ": transfer size (")
            << transfer_bytes
            << "B) does not match any granularity entry "
               "for memory space "
            << space;
        return std::nullopt;
      }

      // Single-element granularity: element-addressable, any stride is legal.
      if (matched_gran_bytes == *elem_bytes) return true;
    }

    // Multi-element transfer: requires contiguous stride.
    // A single element is always legal regardless of stride.
    if (total_elems == 1 || stride_val == 1) return true;  // legal

    // Non-contiguous multi-element transfer, correctable.
    return false;
  }

  /// Classifies a single ktdf.data_transfer and returns a TransferStep.
  /// FIFO sources are returned with all flags false; they are handled once
  /// alignment_factor is known. Returns nullopt on hard failure (error already
  /// emitted). When `inherited_strides` is non-empty it is forwarded to
  /// checkMemRef.
  std::optional<TransferStep> classifyTransferStep(
      mlir::ktdf::DataTransferOp dt,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    TransferStep ts{};
    ts.transfer = dt;

    // Check a memref operand; returns false on hard failure (error emitted).
    auto check_operand =
        [&](mlir::Value operand, llvm::StringRef side,
            std::optional<llvm::SmallVector<int64_t>> sizes) -> bool {
      if (!sizes) {
        dt->emitError(PASS_NAME ": ") << side
                                      << " transfer has dynamic sizes; "
                                         "static sizes are required";
        return false;
      }
      auto memref = mlir::cast<mlir::MemRefType>(operand.getType());
      auto result =
          checkMemRef(dt, memref, *sizes, resource_kinds, inherited_strides);
      if (!result.has_value()) return false;
      if (!*result) ts.is_illegal = true;
      return true;
    };

    if (dt.isSourceMemRef() && !check_operand(dt.getSource(), "source",
                                              dt.getStaticSourceSizesArray()))
      return std::nullopt;

    if (dt.isDestMemRef() && !check_operand(dt.getDestination(), "destination",
                                            dt.getStaticDestSizesArray()))
      return std::nullopt;

    return ts;
  }

  /// Analyses a single stage. If it contains a nested ktdf.pipeline, stores a
  /// stub in sa.nested_pipeline for fixPipeline to resolve. Otherwise
  /// classifies every ktdf.data_transfer into sa.transfers. Sets has_illegal if
  /// any transfer is illegal; is_displaced/needs_splat are set later. When
  /// `inherited_strides` is non-empty it is forwarded to classifyTransferStep.
  StageAnalysis analyzeStage(
      mlir::ktdf::StageOp stage,
      const mlir::ktdf_arch::ResourceKinds& resource_kinds, bool& has_illegal,
      llvm::ArrayRef<int64_t> inherited_strides = {}) {
    StageAnalysis sa{};
    sa.stage = stage;

    // Check if the stage contains a nested pipeline.
    mlir::ktdf::PipelineOp nested_pipeline;
    stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp p) {
      nested_pipeline = p;
      return mlir::WalkResult::interrupt();
    });

    if (nested_pipeline) {
      PipelineAnalysis stub{};
      stub.pipeline = nested_pipeline;
      sa.nested_pipeline = std::make_unique<PipelineAnalysis>(std::move(stub));
      // Also find the innermost loop enclosing the nested pipeline — it needs
      // to be collapsed to ub=1 just like leaf stage loops, so that the only
      // column iteration is the loop inserted by insertLoopAroundPipeline.
      stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp forOp) {
        if (!nested_pipeline->isAncestor(forOp)) sa.innermost_loop = forOp;
      });
      return sa;
    }

    // Find the innermost scf.for enclosing the transfers in this stage.
    // PreOrder visits outer-to-inner; no interrupt() means the last
    // assignment wins, leaving innermost_loop as the innermost ForOp.
    stage->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::scf::ForOp forOp) { sa.innermost_loop = forOp; });

    // Leaf stage: collect every data_transfer op.
    stage->walk([&](mlir::ktdf::DataTransferOp dt) {
      auto ts = classifyTransferStep(dt, resource_kinds, inherited_strides);
      if (ts) {
        if (ts->is_illegal) has_illegal = true;
        sa.transfers.push_back(*ts);
      }
    });

    return sa;
  }
};

/// Prints a TransferStep prefixed by `indent`.
static llvm::raw_ostream& printTransferStep(
    llvm::raw_ostream& os, const DataTransferLegality::TransferStep& ts,
    llvm::StringRef indent) {
  os << indent << "TransferStep{\n";
  os << indent << "  loc=";
  if (ts.transfer)
    os << ts.transfer->getLoc();
  else
    os << "<unset>";
  os << "\n"
     << indent << "  is_illegal=" << ts.is_illegal << "\n"
     << indent << "  is_displaced=" << ts.is_displaced << "\n"
     << indent << "  needs_splat=" << ts.needs_splat << "\n"
     << indent << "}";
  return os;
}

// Forward declaration; printStageAnalysis and printPipelineAnalysis are
// mutually recursive.
static llvm::raw_ostream& printPipelineAnalysis(
    llvm::raw_ostream& os, const DataTransferLegality::PipelineAnalysis& pa,
    llvm::StringRef indent);

/// Prints a StageAnalysis prefixed by `indent`.
static llvm::raw_ostream& printStageAnalysis(
    llvm::raw_ostream& os, const DataTransferLegality::StageAnalysis& sa,
    llvm::StringRef indent) {
  os << indent << "StageAnalysis{stage=";
  if (sa.stage)
    os << sa.stage->getLoc();
  else
    os << "<unset>";
  if (sa.innermost_loop) {
    os << "\n" << indent << "  innermost_loop=";
    sa.innermost_loop->print(os, mlir::OpPrintingFlags().skipRegions());
  }
  if (sa.nested_pipeline) {
    os << ", nested_pipeline=\n";
    printPipelineAnalysis(os, *sa.nested_pipeline, (indent + "  ").str());
  } else if (sa.transfers.empty()) {
    os << ", transfers=<empty>";
  } else {
    os << ", transfers=[\n";
    std::string ts_indent = (indent + "    ").str();
    for (const auto& ts : sa.transfers) printTransferStep(os, ts, ts_indent);
    os << "\n" << indent << "]";
  }
  os << "}";
  return os;
}

/// Prints a PipelineAnalysis prefixed by `indent`.
static llvm::raw_ostream& printPipelineAnalysis(
    llvm::raw_ostream& os, const DataTransferLegality::PipelineAnalysis& pa,
    llvm::StringRef indent) {
  os << indent << "PipelineAnalysis{\n";
  os << indent << "  pipeline=";
  if (pa.pipeline) {
    auto pipeline =
        pa.pipeline;  // mutable copy, PipelineOp is a pointer wrapper
    os << pipeline->getLoc();
    os << ", stages=" << pipeline.getNumStages();
  } else {
    os << "<unset>";
  }
  os << "\n";
  os << indent << "  alignment_kind="
     << (pa.alignment_kind == DataTransferLegality::AlignmentKind::Widen
             ? "Widen"
             : "Shrink")
     << "\n";
  os << indent << "  align_dim=" << pa.align_dim << "\n";
  os << indent << "  alignment_factor=" << pa.alignment_factor << "\n";
  os << indent << "  stride_elems=" << pa.stride_elems << "\n";
  if (pa.allocs.empty()) {
    os << indent << "  allocs=<none>\n";
  } else {
    for (const auto& alloc : pa.allocs)
      os << indent << "  alloc=" << alloc->getLoc() << "\n";
  }
  if (pa.stages.empty()) {
    os << indent << "  stages=<empty>\n";
  } else {
    os << indent << "  stages=[\n";
    std::string sa_indent = (indent + "    ").str();
    for (const auto& sa : pa.stages) {
      printStageAnalysis(os, sa, sa_indent);
      os << "\n";
    }
    os << indent << "  ]\n";
  }
  os << indent << "}";
  return os;
}

/// Prints a PipelineAnalysis at zero indent for LDBG.
static llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const DataTransferLegality::PipelineAnalysis& pa) {
  return printPipelineAnalysis(os, pa, "");
}

/// Returns true if `val` is a memref located in per-core ct_local memory.
static bool isPerCoreScratchpad(
    mlir::Value val, const scheduler::arch_view::MemoryTree& memory_tree) {
  auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
  if (!mrt) return false;
  auto space = mrt.getMemorySpace();
  if (!space) return false;
  auto kind = mlir::dyn_cast<mlir::ktdf_arch::KindAttr>(space);
  if (!kind) return false;
  return memory_tree.isPerCoreScratchPadMemory(kind);
}

/// Widens any ct_local alloc backing a source or destination of `ts` at
/// pa.align_dim by pa.alignment_factor. Non-ct_local sides are skipped.
static void widenAlloc(const DataTransferLegality::TransferStep& ts,
                       const DataTransferLegality::PipelineAnalysis& pa,
                       const scheduler::arch_view::MemoryTree& memory_tree,
                       mlir::OpBuilder& builder) {
  const int64_t alignment_factor = pa.alignment_factor;

  // DataTransferOp is a pointer wrapper — copy it so we can call non-const
  // accessors without needing to drop the const on the TransferStep.
  mlir::ktdf::DataTransferOp dt = ts.transfer;

  // Tries to widen the alloc backing `val` if it is a ct_local memref owned
  // by this pipeline. Returns early silently for non-ct_local or foreign
  // allocs.
  auto try_widen = [&](mlir::Value val) {
    if (!isPerCoreScratchpad(val, memory_tree)) return;

    // ct_local buffers are exposed as ktdf.private results; follow the result
    // index into the private_yield operands to reach the backing memref.alloc.
    // Track private_op so its declared result type can be updated to match.
    mlir::Value base = val;
    mlir::ktdf::PrivateOp private_op;
    unsigned private_result_number = 0;
    if (auto pop = mlir::dyn_cast_or_null<mlir::ktdf::PrivateOp>(
            base.getDefiningOp())) {
      private_result_number =
          mlir::cast<mlir::OpResult>(base).getResultNumber();
      base = pop.getYieldOp().getOperand(private_result_number);
      private_op = pop;
    }

    auto alloc =
        mlir::dyn_cast_or_null<mlir::memref::AllocOp>(base.getDefiningOp());
    if (!alloc) {
      ts.transfer->emitError(
          "widenAlloc: ct_local operand has no backing AllocOp");
      return;
    }

    // Only widen allocs that belong to this pipeline's ktdf.private region.
    // Allocs from an outer pipeline are already handled and must not be
    // touched.
    if (!llvm::is_contained(pa.allocs, alloc)) return;

    mlir::MemRefType orig_type = alloc.getType();
    llvm::SmallVector<int64_t> new_shape(orig_type.getShape());

    // Convert offset-from-end to an absolute index into the alloc shape.
    int64_t alloc_dim = (int64_t)orig_type.getRank() - 1 - pa.align_dim;
    if (alloc_dim < 0 || alloc_dim >= (int64_t)orig_type.getRank()) {
      ts.transfer->emitError("widenAlloc: align_dim ")
          << alloc_dim << " is out of range for alloc rank "
          << orig_type.getRank();
      return;
    }

    llvm::SmallVector<mlir::Value> new_dynamic_sizes(alloc.getDynamicSizes());

    if (orig_type.getNumDynamicDims() > 0 && !new_dynamic_sizes.empty()) {
      builder.setInsertionPoint(alloc);
      mlir::Value c1 =
          mlir::arith::ConstantIndexOp::create(builder, alloc.getLoc(), 1)
              .getResult();
      new_dynamic_sizes.back() = c1;
    } else {
      if (new_shape[alloc_dim] == new_shape[alloc_dim] * alignment_factor)
        return;

      if (new_shape[0] % alignment_factor != 0) {
        ts.transfer->emitError("widenAlloc: outermost dimension (")
            << new_shape[0] << ") is not divisible by alignment factor "
            << alignment_factor;
        return;
      }
      new_shape[0] = new_shape[0] / alignment_factor;
    }

    int64_t widened_dim = new_shape[alloc_dim] * alignment_factor;
    int64_t innermost_dim = new_shape[(int64_t)orig_type.getRank() - 1];
    if (innermost_dim > 0 && widened_dim % innermost_dim != 0) {
      ts.transfer->emitError("widenAlloc: widened dimension (")
          << widened_dim << ") is not a multiple of innermost dimension ("
          << innermost_dim << ")";
      return;
    }
    new_shape[alloc_dim] = widened_dim;

    // Use the default identity layout - the affine maps on the data_transfer
    // ops encode all access patterns.
    mlir::MemRefType new_type = mlir::MemRefType::get(
        new_shape, orig_type.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, orig_type.getMemorySpace());

    builder.setInsertionPoint(alloc);
    auto new_alloc = mlir::memref::AllocOp::create(builder, alloc.getLoc(),
                                                   new_type, new_dynamic_sizes);

    // Keep the ktdf.private result type in sync; the verifier requires it to
    // match the private_yield operand type and the alloc type.
    if (private_op)
      private_op.getResult(private_result_number).setType(new_type);

    LDBG(1) << "  widenAlloc: " << orig_type << " → " << new_type;
    alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
    alloc.erase();
  };

  // Try both sides: dest-side ct_local (e.g. load into staging buffer) and
  // source-side ct_local (e.g. store out of staging buffer).
  try_widen(dt.getDestination());
  try_widen(dt.getSource());
}

/// Reduces sa.innermost_loop upper bound by pa.alignment_factor after verifying
/// that the loop's total_size operand is divisible by pa.alignment_factor.
static mlir::LogicalResult adjustLoopBounds(
    const DataTransferLegality::StageAnalysis& sa,
    const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  if (!sa.innermost_loop) return mlir::success();

  mlir::scf::ForOp loop = sa.innermost_loop;

  // The upper bound is normally a ktdf.tiling.derive_size result (produced by
  // StageCoarseningPass). When the IR was not tiled it may instead be a bare
  // arith.constant.
  int64_t total_val = -1;
  auto derive = mlir::dyn_cast_or_null<mlir::ktdf::TilingDeriveSizeOp>(
      loop.getUpperBound().getDefiningOp());
  if (derive) {
    // total_size is the constant full trip count for this dimension.
    // tile_sizes are still symbolic reserve_size placeholders at this stage.
    auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
        derive.getTotalSize().getDefiningOp());
    if (!cst) {
      return loop->emitError(
                 PASS_NAME
                 ": ktdf.tiling.derive_size total_size is not a constant "
                 "— cannot verify dimension size against alignment_factor=")
             << pa.alignment_factor;
    }
    total_val = cst.value();

    // After this rewrite the loop walks the whole dimension (total_val /
    // alignment_factor widened transfers) on every iteration of the tile loop
    // that encloses it, and the widened alloc drops the tile-size dimension.
    // That is only correct when one tile is the whole dimension, so pin each
    // placeholder tile size to total_val: TileSizeSelection then has no other
    // legal choice, and the enclosing tile loop runs once.
    for (mlir::Value tile_size : derive.getTileSizes()) {
      auto reserve = mlir::dyn_cast_or_null<mlir::ktdf::TilingReserveSizeOp>(
          tile_size.getDefiningOp());
      if (!reserve) {
        auto fixed = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
            tile_size.getDefiningOp());
        if (fixed && fixed.value() == total_val) continue;
        return loop->emitError(
                   PASS_NAME
                   ": ktdf.tiling.derive_size tile size is neither a "
                   "ktdf.tiling.reserve_size placeholder nor the constant "
                   "whole dimension (")
               << total_val << ") that the widened transfers cover";
      }
      mlir::IntegerAttr whole_dim = builder.getIndexAttr(total_val);
      reserve.setMinValueAttr(whole_dim);
      reserve.setDivisibilityAttr(whole_dim);
    }
  } else {
    auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
        loop.getUpperBound().getDefiningOp());
    if (!cst) {
      return loop->emitError(
                 PASS_NAME
                 ": innermost stage loop upper bound is neither a "
                 "ktdf.tiling.derive_size nor a constant index "
                 "— cannot verify dimension size against alignment_factor=")
             << pa.alignment_factor;
    }
    total_val = cst.value();
  }

  if (pa.alignment_factor <= 0 || total_val % pa.alignment_factor != 0) {
    return loop->emitError(PASS_NAME ": dimension total size (")
           << total_val << ") is not divisible by alignment factor "
           << pa.alignment_factor
           << "; tiling and alignment constraints are inconsistent";
  }

  // Each widened transfer now covers alignment_factor times as much data per
  // iteration, so the loop only needs total_val / alignment_factor iterations.
  int64_t new_ub = total_val / pa.alignment_factor;
  builder.setInsertionPoint(loop);
  mlir::Value new_ub_val =
      mlir::arith::ConstantIndexOp::create(builder, loop.getLoc(), new_ub)
          .getResult();
  loop.setUpperBound(new_ub_val);

  LDBG(1) << "  adjustLoopBounds: total_size=" << total_val
          << " / alignment_factor=" << pa.alignment_factor
          << " → new ub=" << new_ub << " at " << loop.getLoc();
  return mlir::success();
}

/// Returns the ktdp.construct_memory_view that `val` is a view of, looking
/// through memref.cast, memref.memory_space_cast and memref.reinterpret_cast.
/// Returns a null op when `val` does not come from one.
static mlir::ktdp::ConstructMemoryViewOp findConstructMemoryView(
    mlir::Value val) {
  mlir::Value cursor = val;
  while (mlir::Operation* def_op = cursor.getDefiningOp()) {
    if (auto construct =
            mlir::dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(def_op))
      return construct;
    if (auto cast_op = mlir::dyn_cast<mlir::memref::CastOp>(def_op))
      cursor = cast_op.getSource();
    else if (auto mscast_op =
                 mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(def_op))
      cursor = mscast_op.getSource();
    else if (auto ricast_op =
                 mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(def_op))
      cursor = ricast_op.getSource();
    else
      break;
  }
  return {};
}

/// Builds a second view of the memory that the global operand `memref` of
/// `op` addresses, with its dimensions permuted: dimension i of the new view is
/// dimension order[i] of `memref` (sizes, strides and coordinate set alike).
/// The new ktdp.construct_memory_view is a clone of the one `memref` comes
/// from; it is cast to `memref`'s memory space and layout offset, so it can
/// replace `memref` as an operand of `op`. Other users of the original view
/// keep it. `memref` must come from a ktdp.construct_memory_view with static
/// sizes and strides equal to the operand's own; otherwise fails with a
/// diagnostic naming `purpose`, the reason for the permutation.
static mlir::FailureOr<mlir::Value> buildPermutedGlobalView(
    mlir::ktdf::DataTransferOp op, mlir::Value memref,
    llvm::ArrayRef<int64_t> order, llvm::StringRef purpose,
    mlir::OpBuilder& builder) {
  auto operand_type = mlir::cast<mlir::MemRefType>(memref.getType());
  mlir::ktdp::ConstructMemoryViewOp construct = findConstructMemoryView(memref);
  llvm::SmallVector<int64_t> strides;
  int64_t operand_offset = 0;
  if (!construct ||
      mlir::failed(operand_type.getStridesAndOffset(strides, operand_offset))) {
    return op->emitError(PASS_NAME ": cannot permute a global memory view ")
           << operand_type
           << " that is not a ktdp.construct_memory_view with a strided "
              "layout, to "
           << purpose;
  }
  const int64_t rank = (int64_t)strides.size();
  if ((int64_t)order.size() != rank) {
    return op->emitError(PASS_NAME ": permutation of rank ")
           << order.size() << " for a view of rank " << rank;
  }
  if (!construct.getSizes().empty() || !construct.getStrides().empty() ||
      llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
    return op->emitError(PASS_NAME
                         ": cannot permute a global memory view with dynamic "
                         "sizes or strides, to ")
           << purpose << "; got " << construct.getResult().getType();
  }
  // The operand's own layout is what the transfer addresses; the construct is
  // only rebuilt from it, so the two must agree.
  if (!llvm::equal(strides, construct.getStaticStrides())) {
    return op->emitError(PASS_NAME
                         ": transfer operand strides differ from those of its "
                         "ktdp.construct_memory_view (a reinterpret_cast in "
                         "between changes them); cannot permute ")
           << operand_type << " to " << purpose;
  }

  mlir::MLIRContext* ctx = op->getContext();
  auto construct_type =
      mlir::cast<mlir::MemRefType>(construct.getResult().getType());
  llvm::ArrayRef<int64_t> shape = construct_type.getShape();
  llvm::SmallVector<int64_t> new_shape, new_strides;
  llvm::SmallVector<mlir::AffineExpr> new_dim_of_old(rank);
  for (int64_t i = 0; i < rank; ++i) {
    new_shape.push_back(shape[order[i]]);
    new_strides.push_back(strides[order[i]]);
    new_dim_of_old[order[i]] = mlir::getAffineDimExpr(i, ctx);
  }
  mlir::IntegerSet old_set = construct.getCoordinateSet().getValue();
  mlir::IntegerSet new_set = old_set.replaceDimsAndSymbols(
      new_dim_of_old, /*symReplacements=*/{}, rank, old_set.getNumSymbols());

  auto offset_of = [](mlir::MemRefType type) {
    llvm::SmallVector<int64_t> unused;
    int64_t offset = 0;
    if (mlir::failed(type.getStridesAndOffset(unused, offset)))
      return int64_t{0};
    return offset;
  };
  auto reordered_type = [&](mlir::MemRefType like) {
    return mlir::MemRefType::get(
        new_shape, like.getElementType(),
        mlir::StridedLayoutAttr::get(ctx, offset_of(like), new_strides),
        like.getMemorySpace());
  };

  builder.setInsertionPointAfter(construct);
  auto new_construct =
      mlir::cast<mlir::ktdp::ConstructMemoryViewOp>(builder.clone(*construct));
  new_construct.setStaticSizesAttr(
      mlir::DenseI64ArrayAttr::get(ctx, new_shape));
  new_construct.setStaticStridesAttr(
      mlir::DenseI64ArrayAttr::get(ctx, new_strides));
  new_construct.setCoordinateSetAttr(mlir::IntegerSetAttr::get(new_set));
  new_construct.getResult().setType(reordered_type(construct_type));

  // Rebuild the operand's memory space and layout offset on the new view.
  builder.setInsertionPoint(op);
  mlir::Value reordered = new_construct.getResult();
  auto in_operand_space =
      mlir::MemRefType::get(new_shape, construct_type.getElementType(),
                            reordered_type(construct_type).getLayout(),
                            operand_type.getMemorySpace());
  if (in_operand_space.getMemorySpace() != construct_type.getMemorySpace())
    reordered = mlir::memref::MemorySpaceCastOp::create(
                    builder, op.getLoc(), in_operand_space, reordered)
                    .getResult();
  mlir::MemRefType final_type = reordered_type(operand_type);
  if (reordered.getType() != final_type)
    reordered = mlir::memref::CastOp::create(builder, op.getLoc(), final_type,
                                             reordered)
                    .getResult();
  LDBG(1) << "  buildPermutedGlobalView: " << construct_type << " → "
          << final_type << " at " << op.getLoc();
  return reordered;
}

/// Re-expresses a global memory view whose strides do not decrease from the
/// outermost dimension inwards, for one side of the transfer `op`.
///
/// Example: sizes [2, 64, 64] with strides [64, 128, 1] (the [64, 128]
/// row-major tensor seen as two 64x64 blocks: each stick is contiguous, and the
/// sticks of the two blocks interleave). The device's address generator derives
/// each dimension's extent from the ratio of neighbouring strides, so it
/// rejects such a layout. This builds a second view of the same memory with the
/// dimensions in decreasing-stride order (sizes [64, 2, 64], strides
/// [128, 64, 1]), and permutes `map`'s results and `sizes` the same way, so the
/// transfer still addresses exactly the same elements. `memref`, `map` and
/// `sizes` are updated in place; other users of the original view keep it.
///
/// No-op for scratchpad operands, for views not built by a
/// ktdp.construct_memory_view, and for views already in decreasing order.
/// Fails, with a diagnostic, on a view with dynamic sizes or strides that
/// would need reordering.
static mlir::LogicalResult orderGlobalViewByStride(
    mlir::ktdf::DataTransferOp op, mlir::Value& memref, mlir::AffineMap& map,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  auto operand_type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!operand_type || isPerCoreScratchpad(memref, memory_tree))
    return mlir::success();
  if (!findConstructMemoryView(memref)) return mlir::success();
  llvm::SmallVector<int64_t> strides;
  int64_t operand_offset = 0;
  if (mlir::failed(operand_type.getStridesAndOffset(strides, operand_offset)))
    return mlir::success();
  const int64_t rank = (int64_t)strides.size();

  // order[i] is the original dimension that becomes dimension i.
  llvm::SmallVector<int64_t> order(rank);
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(
      order, [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });
  if (llvm::equal(order, llvm::seq<int64_t>(0, rank))) return mlir::success();

  auto reordered = buildPermutedGlobalView(
      op, memref, order, "put it into decreasing-stride order", builder);
  if (mlir::failed(reordered)) return mlir::failure();

  // The view's dimensions moved, so the transfer's subscripts and sizes move
  // with them: it still addresses exactly the same elements.
  mlir::MLIRContext* ctx = op->getContext();
  if (!map) map = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  llvm::SmallVector<mlir::AffineExpr> new_results;
  llvm::SmallVector<mlir::OpFoldResult> new_sizes;
  for (int64_t i = 0; i < rank; ++i) {
    new_results.push_back(map.getResult(order[i]));
    new_sizes.push_back(sizes[order[i]]);
  }
  map = mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), new_results,
                             ctx);
  sizes.assign(new_sizes.begin(), new_sizes.end());
  memref = *reordered;
  return mlir::success();
}

/// Rebuilds `op` with every global view whose strides do not decrease
/// re-expressed by orderGlobalViewByStride. This covers transfers no other
/// rewrite touched: e.g. a copy that only permutes outer dims, sizes
/// [64, 64, 64] with input strides [64, 4096, 1], is per-stick legal, so its
/// pipeline needs no widening, yet the device compiler rejects the
/// non-decreasing layout. A no-op when neither side needs reordering.
static mlir::LogicalResult reorderGlobalViewsOf(
    mlir::ktdf::DataTransferOp op,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  mlir::Value src = op.getSource(), dst = op.getDestination();
  mlir::AffineMap src_map = op.isSourceMemRef()
                                ? op.getSourceMapAttr().getValue()
                                : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};
  auto src_sizes = op.getMixedSourceSizes();
  auto dst_sizes = op.getMixedDestSizes();
  if (op.isSourceMemRef() &&
      mlir::failed(orderGlobalViewByStride(op, src, src_map, src_sizes,
                                           memory_tree, builder)))
    return mlir::failure();
  if (op.isDestMemRef() &&
      mlir::failed(orderGlobalViewByStride(op, dst, dst_map, dst_sizes,
                                           memory_tree, builder)))
    return mlir::failure();
  if (src == op.getSource() && dst == op.getDestination())
    return mlir::success();

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), src, src_map, op.getSourceIndices(), src_sizes,
      dst, dst_map, op.getDestIndices(), dst_sizes);
  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());
  op.erase();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Row-dimension normalization
//===----------------------------------------------------------------------===//
//
// A copy pipeline addresses its global views through one shared index space:
// the load and the store use the same index tuple, and each index comes from
// one tile-loop chain, i.e. a ktdf.tiling.reserve_size tile size T and the
// loops built on it:
//
//   %T = ktdf.tiling.reserve_size ...
//   scf.for %tile = 0 to ceildivui(N, %T)              // N = the dim's extent
//     %n = ktdf.tiling.derive_size [%tile : %T], total_size = N
//     ... scf.for %e = 0 to %n
//           %i = ktdf.tiling.linearize_index [%tile : %T], [%e : 1]
//           ktdf.data_transfer from %view[.., %i, ..]
//
// Widening a re-stickify load (see rewriteTransferShape and adjustLoopBounds)
// needs the source's stride-1 dim, the one each input stick runs along, at the
// row position: the second-to-last dim, whose chain is the innermost stage
// loop and the last tile dim of the scratchpad buffers. A producer may order
// the shared dims differently, e.g. a [128, 64, 128] fp16 transpose(1, 2) in
// torch-spyre's layout comes as sizes [1, 2, 64, 128, 64] with input strides
// [1048576, 8192, 1, 64, 16384]: stride 1 is dim 2, the row position is dim 3.
//
// Relabeling the two dims fixes that without moving any loop or buffer: swap
// dims k and r of every global view, and swap the extents N of their two
// chains. The chain that walked dim k now walks the new dim k, i.e. the old
// dim r, and the reverse. Every load and store applies the same relabeling,
// so each loop iteration still pairs the same input element with the same
// output element.

namespace row_dim {

/// The tile-loop chain behind index `position` of a memref endpoint: the
/// ktdf.tiling.reserve_size result that is the tile-size stride of the
/// ktdf.tiling.linearize_index producing that index. Null when the index is
/// not a plain linearize_index result (e.g. a constant for a size-1 dim).
static mlir::Value tileChainOf(mlir::AffineMap map, mlir::ValueRange indices,
                               int64_t position) {
  if (!map || position >= (int64_t)map.getNumResults()) return {};
  auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(position));
  if (!dim || dim.getPosition() >= indices.size()) return {};
  auto linearize = indices[dim.getPosition()]
                       .getDefiningOp<mlir::ktdf::TilingLinearizeIndexOp>();
  if (!linearize) return {};
  mlir::Value chain;
  for (mlir::Value stride : linearize.getStrides()) {
    if (!stride.getDefiningOp<mlir::ktdf::TilingReserveSizeOp>()) continue;
    if (chain) return {};  // two tiled loops feed one index
    chain = stride;
  }
  return chain;
}

/// One global (non-scratchpad) memref side of a transfer in the pipeline.
struct Endpoint {
  mlir::ktdf::DataTransferOp op;
  bool is_source;

  // DataTransferOp is a pointer wrapper with non-const accessors, so each
  // getter works on a copy.
  mlir::Value memref() const {
    mlir::ktdf::DataTransferOp dt = op;
    return is_source ? dt.getSource() : dt.getDestination();
  }
  mlir::AffineMap map() const {
    mlir::ktdf::DataTransferOp dt = op;
    mlir::AffineMapAttr attr =
        is_source ? dt.getSourceMapAttr() : dt.getDestMapAttr();
    return attr ? attr.getValue() : mlir::AffineMap{};
  }
  mlir::ValueRange indices() const {
    mlir::ktdf::DataTransferOp dt = op;
    return is_source ? dt.getSourceIndices() : dt.getDestIndices();
  }
};

/// The ops that hold a chain's extent N: every ktdf.tiling.derive_size with
/// the chain as its only tile size (total_size = N) and every
/// arith.ceildivui N, chain (the tile loop's trip count).
struct ChainExtentUses {
  llvm::SmallVector<mlir::ktdf::TilingDeriveSizeOp> derives;
  llvm::SmallVector<mlir::arith::CeilDivUIOp> ceildivs;
};

/// Collects the uses of `chain` that hold its extent, checking that each one
/// is the constant `extent`. Fails with a diagnostic at `at` for any use this
/// relabeling does not know how to keep consistent.
static mlir::FailureOr<ChainExtentUses> collectExtentUses(
    mlir::Value chain, int64_t extent, int64_t dim, mlir::Operation* at) {
  auto constant_is = [&](mlir::Value v) {
    auto cst = v.getDefiningOp<mlir::arith::ConstantIndexOp>();
    return cst && cst.value() == extent;
  };
  ChainExtentUses uses;
  for (mlir::OpOperand& use : chain.getUses()) {
    mlir::Operation* user = use.getOwner();
    if (mlir::isa<mlir::ktdf::TilingLinearizeIndexOp, mlir::memref::AllocOp>(
            user))
      continue;  // tile-size uses; the extent lives elsewhere
    if (auto derive = mlir::dyn_cast<mlir::ktdf::TilingDeriveSizeOp>(user)) {
      if (derive.getTileSizes().size() == 1 &&
          constant_is(derive.getTotalSize())) {
        uses.derives.push_back(derive);
        continue;
      }
    } else if (auto ceildiv = mlir::dyn_cast<mlir::arith::CeilDivUIOp>(user)) {
      if (ceildiv.getRhs() == chain && constant_is(ceildiv.getLhs())) {
        uses.ceildivs.push_back(ceildiv);
        continue;
      }
    }
    return at->emitError(PASS_NAME ": cannot move the source's stride-1 dim "
                                   "to the row position: the tile loop of "
                                   "dim ")
           << dim << " (extent " << extent
           << ") has a use that does not state that extent as a constant "
              "ktdf.tiling.derive_size total or arith.ceildivui dividend: "
           << *user;
  }
  return uses;
}

/// Sets every extent use in `uses` to the constant `extent`.
static void setExtent(const ChainExtentUses& uses, int64_t extent,
                      mlir::OpBuilder& builder) {
  for (mlir::ktdf::TilingDeriveSizeOp derive : uses.derives) {
    builder.setInsertionPoint(derive);
    mlir::Value cst = mlir::arith::ConstantIndexOp::create(
        builder, derive.getLoc(), extent);
    derive.getTotalSizeMutable().assign(cst);
  }
  for (mlir::arith::CeilDivUIOp ceildiv : uses.ceildivs) {
    builder.setInsertionPoint(ceildiv);
    mlir::Value cst = mlir::arith::ConstantIndexOp::create(
        builder, ceildiv.getLoc(), extent);
    ceildiv.getLhsMutable().assign(cst);
  }
}

}  // namespace row_dim

/// Relabels the shared index space of `pipeline` so that a re-stickify
/// source's stride-1 dim sits at the row position (the second-to-last dim),
/// where widening expects it (see "Row-dimension normalization" above).
///
/// Applies when the pipeline has a global source view whose last two strides
/// are both not 1 and exactly one earlier dim k of extent > 1 has stride 1.
/// With r the row position, it swaps dims k and r of every global view the
/// pipeline addresses (sizes, strides, coordinate set, and the transfer's
/// sizes at those positions; subscripts are unchanged), and swaps the extents
/// of the tile-loop chains indexing positions k and r.
///
/// Example: input sizes [1, 2, 64, 128, 64], strides [1048576, 8192, 1, 64,
/// 16384]; output strides [1048576, 524288, 8192, 64, 1]; chains of extents 64
/// (position 2) and 128 (position 3). After: input sizes [1, 2, 128, 64, 64],
/// strides [1048576, 8192, 64, 1, 16384]; output strides [1048576, 524288, 64,
/// 8192, 1]; the chain at position 2 has extent 128, the one at position 3
/// extent 64.
///
/// A no-op for every other pipeline. Fails, with a diagnostic, when the
/// relabeling applies but the pipeline does not have the shape it relies on:
/// every global endpoint of one rank and the same extents at k and r, both
/// positions indexed by one tile-loop chain each, the same chain across
/// endpoints, the two chains' tile-size constraints equal, and each chain's
/// extent stated only as constants.
static mlir::LogicalResult moveStrideOneDimToRowPosition(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  using namespace row_dim;
  llvm::SmallVector<Endpoint> endpoints;
  pipeline.walk([&](mlir::ktdf::DataTransferOp dt) {
    for (bool is_source : {true, false}) {
      Endpoint e{dt, is_source};
      if (mlir::isa<mlir::MemRefType>(e.memref().getType()) &&
          !isPerCoreScratchpad(e.memref(), memory_tree))
        endpoints.push_back(e);
    }
  });

  // Find the source whose stride-1 dim is neither of the last two.
  int64_t rank = 0, k = -1;
  mlir::ktdf::DataTransferOp anchor;
  for (const Endpoint& e : endpoints) {
    if (!e.is_source) continue;
    auto type = mlir::cast<mlir::MemRefType>(e.memref().getType());
    llvm::SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (mlir::failed(type.getStridesAndOffset(strides, offset))) continue;
    const int64_t n = (int64_t)strides.size();
    if (n < 3 || strides[n - 1] == 1 || strides[n - 2] == 1) continue;
    llvm::SmallVector<int64_t> unit_dims;
    for (int64_t d = 0; d < n - 2; ++d)
      if (strides[d] == 1 && type.getShape()[d] != 1) unit_dims.push_back(d);
    if (unit_dims.size() != 1) continue;
    rank = n;
    k = unit_dims.front();
    anchor = e.op;
    break;
  }
  if (k < 0) return mlir::success();
  const int64_t r = rank - 2;

  // Every global endpoint must agree on rank, on the extents at k and r, and
  // on the chain indexing each.
  auto anchor_type = mlir::cast<mlir::MemRefType>(anchor.getSource().getType());
  const int64_t extent_k = anchor_type.getShape()[k];
  const int64_t extent_r = anchor_type.getShape()[r];
  mlir::Value chain_k, chain_r;
  for (const Endpoint& e : endpoints) {
    auto type = mlir::cast<mlir::MemRefType>(e.memref().getType());
    if (type.getRank() != rank || type.getShape()[k] != extent_k ||
        type.getShape()[r] != extent_r) {
      return e.op->emitError(PASS_NAME
                             ": cannot move the source's stride-1 dim ")
             << k << " to the row position " << r
             << ": the pipeline's global views do not share one index space ("
             << anchor_type << " and " << type << ")";
    }
    mlir::Value at_k = tileChainOf(e.map(), e.indices(), k);
    mlir::Value at_r = tileChainOf(e.map(), e.indices(), r);
    if (!at_k || !at_r) {
      return e.op->emitError(PASS_NAME
                             ": cannot move the source's stride-1 dim ")
             << k << " to the row position " << r << ": dim "
             << (at_k ? r : k)
             << " is not indexed by a ktdf.tiling.linearize_index over one "
                "ktdf.tiling.reserve_size tile loop";
    }
    if (!chain_k) {
      chain_k = at_k;
      chain_r = at_r;
    } else if (at_k != chain_k || at_r != chain_r) {
      return e.op->emitError(PASS_NAME
                             ": cannot move the source's stride-1 dim ")
             << k << " to the row position " << r
             << ": the pipeline's transfers index those dims through "
                "different tile loops";
    }
  }
  if (chain_k == chain_r) {
    return anchor->emitError(PASS_NAME ": dims ")
           << k << " and " << r << " are indexed by the same tile loop";
  }
  auto reserve_k = chain_k.getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
  auto reserve_r = chain_r.getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
  if (reserve_k.getMinValueAttr() != reserve_r.getMinValueAttr() ||
      reserve_k.getDivisibilityAttr() != reserve_r.getDivisibilityAttr()) {
    return anchor->emitError(PASS_NAME
                             ": cannot move the source's stride-1 dim ")
           << k << " to the row position " << r
           << ": their tile loops have different tile-size constraints ("
           << *reserve_k << " and " << *reserve_r << ")";
  }
  auto uses_k = collectExtentUses(chain_k, extent_k, k, anchor);
  auto uses_r = collectExtentUses(chain_r, extent_r, r, anchor);
  if (mlir::failed(uses_k) || mlir::failed(uses_r)) return mlir::failure();

  // Relabel: views and transfer sizes swap dims k and r; chains swap extents.
  llvm::SmallVector<int64_t> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::swap(order[k], order[r]);
  llvm::SmallVector<mlir::ktdf::DataTransferOp> ops;
  for (const Endpoint& e : endpoints)
    if (!llvm::is_contained(ops, e.op)) ops.push_back(e.op);
  for (mlir::ktdf::DataTransferOp op : ops) {
    mlir::Value src = op.getSource(), dst = op.getDestination();
    auto src_sizes = op.getMixedSourceSizes();
    auto dst_sizes = op.getMixedDestSizes();
    for (const Endpoint& e : endpoints) {
      if (e.op != op) continue;
      auto permuted = buildPermutedGlobalView(
          op, e.memref(), order,
          "move the source's stride-1 dim to the row position", builder);
      if (mlir::failed(permuted)) return mlir::failure();
      (e.is_source ? src : dst) = *permuted;
      auto& sizes = e.is_source ? src_sizes : dst_sizes;
      std::swap(sizes[k], sizes[r]);
    }
    mlir::AffineMap src_map = op.isSourceMemRef()
                                  ? op.getSourceMapAttr().getValue()
                                  : mlir::AffineMap{};
    mlir::AffineMap dst_map =
        op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};
    builder.setInsertionPoint(op);
    auto new_op = mlir::ktdf::DataTransferOp::create(
        builder, op.getLoc(), src, src_map, op.getSourceIndices(), src_sizes,
        dst, dst_map, op.getDestIndices(), dst_sizes);
    for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
      new_op->setDiscardableAttr(attr.getName(), attr.getValue());
    op.erase();
  }
  setExtent(*uses_k, extent_r, builder);
  setExtent(*uses_r, extent_k, builder);

  LDBG(1) << "  moveStrideOneDimToRowPosition: swapped dims " << k << " and "
          << r << " (extents " << extent_k << ", " << extent_r << ") at "
          << pipeline.getLoc();
  return mlir::success();
}

/// Replaces ts.transfer with a new DataTransferOp with pa.align_dim widened.
///
/// A global (non-ct_local) memory view whose innermost stride is not 1 is the
/// source of a re-stickify: its sticks run across the last dimension. Its
/// trailing R x C block (sizes [.., R, C], strides [.., 1, R]) is one
/// contiguous column-major run of R*C elements. The widened transfer copies
/// that run into the scratchpad verbatim, as a row-major [R][C] tile, so the
/// view is re-expressed with strides [.., C, 1]; the re-stickify then
/// rearranges the tile on chip. For a square tile this swaps the last two
/// strides (e.g. [4096, 1, 64] becomes [4096, 64, 1]). A view whose innermost
/// stride is already 1 keeps its strides, including non-row-major ones such as
/// [64, 128, 1] (sticks interleaved across the leading dimensions): the
/// transfer lowering addresses each stick through them. Fails, with a
/// diagnostic, on a strided view that is neither.
static mlir::LogicalResult rewriteTransferShape(
    const DataTransferLegality::TransferStep& ts,
    const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  // Convert offset-from-end to absolute indices for src and dst size vectors.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  int64_t src_rank = (int64_t)new_src_sizes.size();
  int64_t dst_rank = (int64_t)new_dst_sizes.size();
  int64_t src_dim = src_rank - 1 - pa.align_dim;
  int64_t dst_dim = dst_rank - 1 - pa.align_dim;

  auto get_scaled = [&](mlir::OpFoldResult ofr) -> mlir::OpFoldResult {
    int64_t existing =
        mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(ofr))
            .getInt();
    return mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                  existing * pa.alignment_factor);
  };

  if (src_dim >= 0 && src_dim < src_rank)
    new_src_sizes[src_dim] = get_scaled(new_src_sizes[src_dim]);
  if (dst_dim >= 0 && dst_dim < dst_rank)
    new_dst_sizes[dst_dim] = get_scaled(new_dst_sizes[dst_dim]);

  mlir::AffineMap src_map = op.isSourceMemRef()
                                ? op.getSourceMapAttr().getValue()
                                : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  // Re-express a re-stickify source view by walking back to its
  // ktdp.construct_memory_view and swapping the last two entries of its
  // static_strides attribute in place (see the function comment). `is_source`
  // says which side of the transfer `val` is: only a load can feed the on-chip
  // transpose.
  auto fix_construct_strides = [&](mlir::Value val,
                                   bool is_source) -> mlir::LogicalResult {
    auto mrt = mlir::dyn_cast<mlir::MemRefType>(val.getType());
    if (!mrt) return mlir::success();
    if (isPerCoreScratchpad(val, memory_tree)) return mlir::success();
    if (!mlir::isa<mlir::StridedLayoutAttr>(mrt.getLayout()))
      return mlir::success();

    mlir::ktdp::ConstructMemoryViewOp construct = findConstructMemoryView(val);
    if (!construct) return mlir::success();

    auto orig_mrt =
        mlir::cast<mlir::MemRefType>(construct.getResult().getType());
    auto shape = orig_mrt.getShape();
    const int64_t rank = (int64_t)shape.size();
    llvm::SmallVector<int64_t> strides(construct.getStaticStrides());
    if ((int64_t)strides.size() != rank ||
        llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
      return op->emitError(PASS_NAME
                           ": memory view strides must be static to widen "
                           "this transfer; got ")
             << orig_mrt;
    }
    // Sticks already contiguous: the strides are addressed as written.
    if (strides.back() == 1) return mlir::success();

    if (!is_source) {
      return op->emitError(PASS_NAME
                           ": destination memory view has innermost stride ")
             << strides.back()
             << "; only a source view can be re-stuck (the on-chip transpose "
                "follows the load), got "
             << orig_mrt;
    }
    // A square tile whose columns are S >= R apart (strides [.., 1, S]) is
    // read column by column: swapping the last two strides gives [.., S, 1],
    // one scratchpad row per source column, which is the [c][t] order the
    // per-element re-stick reads for a square tile. A non-square tile needs
    // the columns to be one contiguous run (S == R), read as row-major
    // [R][C], because its rows are split across column-sized scratchpad rows.
    const bool square = rank >= 2 && shape[rank - 1] == shape[rank - 2];
    if (rank < 2 || strides[rank - 2] != 1 ||
        (square ? strides[rank - 1] < shape[rank - 2]
                : strides[rank - 1] != shape[rank - 2])) {
      return op->emitError(PASS_NAME
                           ": source memory view's trailing tile is neither a "
                           "square tile with column pitch >= its rows "
                           "(strides [.., 1, S], S >= R) nor one contiguous "
                           "column-major block (strides [.., 1, R] for "
                           "trailing sizes [.., R, C]); got ")
             << orig_mrt;
    }
    if (square) {
      std::swap(strides[rank - 2], strides[rank - 1]);
    } else {
      strides[rank - 2] = shape[rank - 1];
      strides[rank - 1] = 1;
    }

    LDBG(1) << "  fix_construct_strides: reading the trailing column-major "
               "block as row-major on "
            << construct->getLoc();
    construct.setStaticStridesAttr(mlir::DenseI64ArrayAttr::get(ctx, strides));

    // Update the construct op's result type to carry the new row-major layout,
    // then propagate the updated type forward through all downstream view-like
    // ops (memory_space_cast, cast, reinterpret_cast) so the strided layout
    // doesn't linger in their result types.
    auto new_layout = mlir::StridedLayoutAttr::get(
        ctx, /*offset=*/mlir::ShapedType::kDynamic, strides);
    auto new_construct_type =
        mlir::MemRefType::get(shape, orig_mrt.getElementType(), new_layout,
                              orig_mrt.getMemorySpace());
    construct.getResult().setType(new_construct_type);

    // Walk forward through uses, updating each view-like op's result type.
    llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
    while (!worklist.empty()) {
      mlir::Value v = worklist.pop_back_val();
      for (mlir::Operation* user : v.getUsers()) {
        mlir::Value new_val;
        mlir::MemRefType src_type =
            mlir::dyn_cast<mlir::MemRefType>(v.getType());
        if (!src_type) continue;

        if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(user)) {
          // Propagate the new layout into the cast result type.
          auto res_type =
              mlir::dyn_cast<mlir::MemRefType>(castOp.getResult().getType());
          if (!res_type) continue;
          auto updated = mlir::MemRefType::get(
              res_type.getShape(), res_type.getElementType(),
              src_type.getLayout(), res_type.getMemorySpace());
          castOp.getResult().setType(updated);
          new_val = castOp.getResult();
        } else if (auto mscastOp =
                       mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(user)) {
          auto res_type =
              mlir::dyn_cast<mlir::MemRefType>(mscastOp.getResult().getType());
          if (!res_type) continue;
          auto updated = mlir::MemRefType::get(
              res_type.getShape(), res_type.getElementType(),
              src_type.getLayout(), res_type.getMemorySpace());
          mscastOp.getResult().setType(updated);
          new_val = mscastOp.getResult();
        } else {
          continue;
        }
        worklist.push_back(new_val);
      }
    }
    return mlir::success();
  };

  if (mlir::failed(fix_construct_strides(op.getSource(), /*is_source=*/true)) ||
      mlir::failed(
          fix_construct_strides(op.getDestination(), /*is_source=*/false)))
    return mlir::failure();

  mlir::Value new_src = op.getSource();
  mlir::Value new_dst = op.getDestination();
  if (op.isSourceMemRef() &&
      mlir::failed(orderGlobalViewByStride(op, new_src, src_map, new_src_sizes,
                                           memory_tree, builder)))
    return mlir::failure();
  if (op.isDestMemRef() &&
      mlir::failed(orderGlobalViewByStride(op, new_dst, dst_map, new_dst_sizes,
                                           memory_tree, builder)))
    return mlir::failure();

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), new_src, src_map, op.getSourceIndices(),
      new_src_sizes, new_dst, dst_map, op.getDestIndices(), new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  LDBG(1) << "  rewriteTransferShape: scaled dim " << pa.align_dim
          << " by factor " << pa.alignment_factor << ": " << new_op;
  op.erase();
  return mlir::success();
}

/// Replaces ts.transfer with a one-element DataTransferOp, with transfer_mode
/// set to `mode` ("splat") or empty for FIFO→memref.
/// The two innermost indices on any ct_local memref operand are replaced with
/// the row and column IVs from the nested scf.for loops inserted by
/// insertLoopAroundPipeline, giving direct [0, 0, %row, %col] addressing.
static void rewriteTransferShrink(
    const DataTransferLegality::TransferStep& ts,
    const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree, llvm::StringRef mode,
    mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  mlir::OpFoldResult one = mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1);

  // Both directions move one element. A splat load collapses the source to
  // one element and keeps the FIFO destination at its natural size [E], so the
  // hardware broadcasts that element across the full slot. A FIFO -> memory
  // transfer stores exactly one element: a FIFO-wide destination would be an
  // E-element store at an element address, which overlaps its neighbours and,
  // at a 128-byte store width, is misaligned on the device.
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  for (auto& s : new_src_sizes) s = one;
  if (op.isSourceFifo() && op.isDestMemRef())
    for (auto& size : new_dst_sizes) size = one;

  // Find the two loop IVs inserted by insertLoopAroundPipeline. Walking out
  // past the enclosing ktdf.pipeline(s) we expect to hit the inner scf.for
  // (col) first, then the outer scf.for (row).
  mlir::Value row_iv, col_iv;
  mlir::Operation* cursor = op->getParentOp();
  while (cursor) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(cursor)) {
      cursor = cursor->getParentOp();
      continue;
    }
    if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(cursor)) {
      if (!col_iv) {
        col_iv = for_op.getInductionVar();
        LDBG(1) << "  rewriteTransferShrink: found col_iv at "
                << for_op->getLoc();
      } else {
        row_iv = for_op.getInductionVar();
        LDBG(1) << "  rewriteTransferShrink: found row_iv at "
                << for_op->getLoc();
        break;
      }
    }
    cursor = cursor->getParentOp();
  }
  if (!row_iv || !col_iv)
    LDBG(1) << "  rewriteTransferShrink: row_iv=" << (bool)row_iv
            << " col_iv=" << (bool)col_iv << " — apply_row_col will no-op";

  // For ct_local memref operands, build a fresh affine map and index list.
  // The load (source) buffer is row-major: index [0, 0, %row, %col].
  // The store (destination) buffer is transposed (column-major): index
  // [0, 0, %col, %row] — row and col are swapped to match the word layout.
  auto apply_row_col = [&](mlir::Value memref_val, mlir::AffineMap& map,
                           llvm::SmallVector<mlir::Value>& indices,
                           bool transpose) {
    if (!row_iv || !col_iv) return;
    if (!isPerCoreScratchpad(memref_val, memory_tree)) return;
    auto mrt = mlir::cast<mlir::MemRefType>(memref_val.getType());

    int64_t rank = (int64_t)mrt.getRank();
    if (rank < 2) return;

    // d0 = first IV, d1 = second IV.
    // For load: (row, col) -> [0,0,row,col].
    // For store (transposed): (col, row) -> [0,0,col,row] i.e. swap the IVs.
    mlir::Value first_iv = transpose ? col_iv : row_iv;
    mlir::Value second_iv = transpose ? row_iv : col_iv;

    llvm::SmallVector<mlir::AffineExpr> results(
        rank, mlir::getAffineConstantExpr(0, ctx));
    results[rank - 2] = mlir::getAffineDimExpr(0, ctx);
    results[rank - 1] = mlir::getAffineDimExpr(1, ctx);
    map = mlir::AffineMap::get(/*dims=*/2, /*syms=*/0, results, ctx);
    indices = {first_iv, second_iv};
  };

  auto new_src_indices = llvm::SmallVector<mlir::Value>(op.getSourceIndices());
  auto new_dst_indices = llvm::SmallVector<mlir::Value>(op.getDestIndices());

  mlir::AffineMap src_map = op.isSourceMemRef()
                                ? op.getSourceMapAttr().getValue()
                                : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  apply_row_col(op.getSource(), src_map, new_src_indices, /*transpose=*/false);
  apply_row_col(op.getDestination(), dst_map, new_dst_indices,
                /*transpose=*/true);

  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), op.getSource(), src_map, new_src_indices,
      new_src_sizes, op.getDestination(), dst_map, new_dst_indices,
      new_dst_sizes);

  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());

  if (!mode.empty())
    new_op->setDiscardableAttr(mlir::StringAttr::get(ctx, "transfer_mode"),
                               mlir::StringAttr::get(ctx, mode));

  LDBG(1) << "  rewriteTransferShrink(" << mode << "): shrunk dim "
          << pa.align_dim << " to 1: " << new_op;
  op.erase();
}

/// Wraps the nested ktdf.pipeline in two nested scf.for loops over [0, bound)
/// so that each element of the widened alloc block can be addressed
/// directly via %row and %col IVs without any index arithmetic.
static void insertLoopAroundPipeline(
    DataTransferLegality::PipelineAnalysis* nested_pa, int64_t bound,
    mlir::OpBuilder& builder) {
  mlir::ktdf::PipelineOp nested_pipeline = nested_pa->pipeline;
  mlir::Location loc = nested_pipeline.getLoc();

  builder.setInsertionPoint(nested_pipeline);

  mlir::Value c0 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value c_bound =
      mlir::arith::ConstantIndexOp::create(builder, loc, bound).getResult();
  mlir::Value c1 =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

  // Outer loop: rows [0, bound)
  auto row_loop = mlir::scf::ForOp::create(builder, loc, c0, c_bound, c1);
  mlir::Block* row_body = row_loop.getBody();

  // Inner loop: cols [0, bound) — inserted inside the row loop body.
  builder.setInsertionPoint(row_body, row_body->getTerminator()->getIterator());
  auto col_loop = mlir::scf::ForOp::create(builder, loc, c0, c_bound, c1);

  // Move the nested pipeline into the col loop body.
  mlir::Block* col_body = col_loop.getBody();
  nested_pipeline->moveBefore(col_body,
                              col_body->getTerminator()->getIterator());

  LDBG(1) << "  insertLoopAroundPipeline: inserted row/col loops (bound="
          << bound << ") around nested pipeline at " << loc;
}

//===----------------------------------------------------------------------===//
// Per-element re-stickify
//===----------------------------------------------------------------------===//
//
// A nested LXLU -> SFP -> LXSU pipeline whose source is read across sticks
// re-sticks one tile: output[.., t, c] = tile element (t, c) of a source tile
// that holds the same R x C tile column-major (element (t, c) at linear
// position c * R + t of the row-major [R][C] scratchpad buffer, which is how
// the widened bulk copy lands a column-major block; see rewriteTransferShape).
//
// It is rebuilt as one element per iteration of loops over the tile:
//   LXLU  one element, broadcast to every lane of the FIFO (transfer_mode
//         "splat"): the load unit has no access that puts one element of a
//         strided column into a chosen lane, only a broadcast of it.
//   SFP   unchanged pass-through.
//   LXSU  one element (a 2-byte store for f16) at [.., t, c]. A FIFO-wide
//         store here would be a 128-byte store at a 2-byte offset, which
//         overlaps neighbouring elements and faults on the device.
// The loops are t_hi < R / C, t_lo < C, c < C, with t = t_hi * C + t_lo, so
// the source element is [c * (R / C) + t_hi][t_lo]: affine, with no floordiv.
// For a square tile R / C = 1 and the t_hi loop is left out. Every leading
// (tile) index of the original transfers is kept, so each tile of a tiled
// buffer is re-stuck at its own position.

namespace restickify {

/// Builds an affine index for one memref endpoint of a new transfer: every
/// leading dim keeps the index the original endpoint had, and the last two
/// dims are `row` and `col`, expressions over the dims standing for `ivs`.
static void buildEndpointIndex(mlir::AffineMap orig_map,
                               mlir::ValueRange orig_indices,
                               mlir::ValueRange ivs, mlir::AffineExpr row,
                               mlir::AffineExpr col, mlir::AffineMap& map,
                               llvm::SmallVector<mlir::Value>& indices) {
  mlir::MLIRContext* ctx = orig_map.getContext();
  const unsigned orig_dims = orig_map.getNumDims();
  const unsigned rank = orig_map.getNumResults();
  llvm::SmallVector<mlir::AffineExpr> results(orig_map.getResults());
  results[rank - 2] = row.shiftDims(ivs.size(), orig_dims);
  results[rank - 1] = col.shiftDims(ivs.size(), orig_dims);
  map = mlir::AffineMap::get(orig_dims + ivs.size(), orig_map.getNumSymbols(),
                             results, ctx);
  indices.assign(orig_indices.begin(), orig_indices.end());
  indices.append(ivs.begin(), ivs.end());
}

static llvm::SmallVector<mlir::OpFoldResult> staticSizes(
    mlir::MLIRContext* ctx, llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  for (int64_t s : sizes)
    result.push_back(mlir::IntegerAttr::get(mlir::IndexType::get(ctx), s));
  return result;
}

/// The load and store transfers of a nested pass-through pipeline.
struct NestedParts {
  mlir::ktdf::DataTransferOp load, store;
};

/// Finds the parts of `pipeline`, which must be a single-transfer LXLU stage,
/// a pass-through compute stage (one read_from_fifo, one write_to_fifo of
/// that value, nothing else) and a single-transfer LXSU stage.
static mlir::FailureOr<NestedParts> findNestedParts(
    mlir::ktdf::PipelineOp pipeline) {
  NestedParts parts;
  bool has_compute = false;
  for (mlir::Operation& op : pipeline.getBody()->getOperations()) {
    auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op);
    if (!stage) continue;
    llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
    stage.walk([&](mlir::ktdf::DataTransferOp t) { transfers.push_back(t); });
    auto reads =
        llvm::to_vector(stage.getBody()->getOps<mlir::ktdf::ReadFromFifoOp>());
    if (transfers.size() == 1 && transfers[0].isDestFifo()) {
      parts.load = transfers[0];
    } else if (transfers.size() == 1 && transfers[0].isSourceFifo()) {
      parts.store = transfers[0];
    } else if (transfers.empty() && reads.size() == 1) {
      auto writes =
          llvm::to_vector(stage.getBody()->getOps<mlir::ktdf::WriteToFifoOp>());
      const auto body_ops =
          std::distance(stage.getBody()->begin(), stage.getBody()->end());
      if (writes.size() != 1 || writes[0].getData() != reads[0].getResult() ||
          body_ops != 2) {
        return stage->emitError(PASS_NAME
                                ": re-stickify needs a pass-through compute "
                                "stage (one read_from_fifo written back "
                                "unchanged); this one computes");
      }
      has_compute = true;
    }
  }
  if (!parts.load || !has_compute || !parts.store) {
    return pipeline->emitError(PASS_NAME
                               ": re-stickify needs a nested pipeline of one "
                               "load, one pass-through compute and one store "
                               "stage");
  }
  return parts;
}

}  // namespace restickify

/// Rewrites the nested pipeline of stage `sa`, which re-sticks one tile (see
/// "Per-element re-stickify" above), into one element per iteration of loops
/// over the tile. Fails with a diagnostic naming what is unsupported.
static mlir::LogicalResult buildPerElementRestickify(
    DataTransferLegality::StageAnalysis& sa, mlir::OpBuilder& builder) {
  using namespace restickify;
  mlir::ktdf::PipelineOp nested = sa.nested_pipeline->pipeline;
  auto found = findNestedParts(nested);
  if (mlir::failed(found)) return mlir::failure();
  NestedParts parts = *found;

  mlir::Value src = parts.load.getSource();
  mlir::Value dst = parts.store.getDestination();
  auto src_type = mlir::cast<mlir::MemRefType>(src.getType());
  auto dst_type = mlir::cast<mlir::MemRefType>(dst.getType());
  const int64_t rank = src_type.getRank();
  if (rank < 2 || dst_type != src_type ||
      mlir::ShapedType::isDynamic(src_type.getShape()[rank - 2]) ||
      mlir::ShapedType::isDynamic(src_type.getShape()[rank - 1])) {
    return parts.load->emitError(PASS_NAME
                                 ": re-stickify needs source and "
                                 "destination buffers of the same "
                                 "type, rank >= 2 and static last "
                                 "two dims, got ")
           << src_type << " and " << dst_type;
  }
  // The tile is [rows][cols]: each output row is cols elements.
  const int64_t rows = src_type.getShape()[rank - 2];
  const int64_t cols = src_type.getShape()[rank - 1];
  if (rows % cols != 0) {
    return parts.load->emitError(PASS_NAME ": re-stickify of a ")
           << rows << "x" << cols
           << " tile is not supported: the row count must be a multiple of "
              "the column count, so one source column spans whole tile rows";
  }
  const int64_t col_spans = rows / cols;
  auto fifo_type = mlir::cast<mlir::ktdf::FifoSlotType>(
      parts.load.getDestination().getType());
  const int64_t fifo_lanes = fifo_type.getNumElements();

  // Loops ([t_hi,] t_lo, c) around the nested pipeline, outermost first.
  mlir::MLIRContext* ctx = builder.getContext();
  mlir::Location loc = nested.getLoc();
  llvm::SmallVector<int64_t> trips;
  if (col_spans > 1) trips.push_back(col_spans);
  trips.append({cols, cols});
  builder.setInsertionPoint(nested);
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value c1 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  llvm::SmallVector<mlir::Value> ivs;
  for (int64_t trip : trips) {
    mlir::Value ub = mlir::arith::ConstantIndexOp::create(builder, loc, trip);
    auto loop = mlir::scf::ForOp::create(builder, loc, c0, ub, c1);
    ivs.push_back(loop.getInductionVar());
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }
  nested->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());

  // Dims d0.. stand for ivs in order: c is the innermost loop.
  const unsigned c_dim = ivs.size() - 1;
  mlir::AffineExpr c = mlir::getAffineDimExpr(c_dim, ctx);
  mlir::AffineExpr t_lo = mlir::getAffineDimExpr(c_dim - 1, ctx);
  mlir::AffineExpr t_hi = col_spans > 1 ? mlir::getAffineDimExpr(0, ctx)
                                        : mlir::getAffineConstantExpr(0, ctx);
  const llvm::SmallVector<int64_t> one_elem(rank, 1);
  mlir::AffineMap map;
  llvm::SmallVector<mlir::Value> indices;

  // LXLU: source element [c * col_spans + t_hi][t_lo], broadcast.
  builder.setInsertionPoint(parts.load);
  buildEndpointIndex(parts.load.getSourceMapAttr().getValue(),
                     parts.load.getSourceIndices(), ivs, c * col_spans + t_hi,
                     t_lo, map, indices);
  auto load = mlir::ktdf::DataTransferOp::create(
      builder, parts.load.getLoc(), src, map, indices,
      staticSizes(ctx, one_elem), parts.load.getDestination(),
      mlir::AffineMap{}, mlir::ValueRange{}, staticSizes(ctx, {fifo_lanes}));
  for (mlir::NamedAttribute attr : parts.load->getDiscardableAttrs())
    load->setDiscardableAttr(attr.getName(), attr.getValue());
  load->setDiscardableAttr(mlir::StringAttr::get(ctx, "transfer_mode"),
                           mlir::StringAttr::get(ctx, "splat"));
  parts.load.erase();

  // LXSU: one element at [t_hi * cols + t_lo][c].
  builder.setInsertionPoint(parts.store);
  buildEndpointIndex(parts.store.getDestMapAttr().getValue(),
                     parts.store.getDestIndices(), ivs, t_hi * cols + t_lo,
                     c, map, indices);
  auto store = mlir::ktdf::DataTransferOp::create(
      builder, parts.store.getLoc(), parts.store.getSource(), mlir::AffineMap{},
      mlir::ValueRange{}, staticSizes(ctx, {fifo_lanes}), dst, map, indices,
      staticSizes(ctx, one_elem));
  for (mlir::NamedAttribute attr : parts.store->getDiscardableAttrs())
    store->setDiscardableAttr(attr.getName(), attr.getValue());
  parts.store.erase();

  LDBG(1) << "  buildPerElementRestickify: rows=" << rows << " cols=" << cols
          << " at " << loc;
  return mlir::success();
}

/// Applies corrective rewrites to the PipelineAnalysis tree: inserts a loop
/// around nested pipelines, then for each leaf stage adjusts the loop bound
/// and rewrites every transfer shape (illegal/displaced → widen; FIFO →
/// splat shrink).
static mlir::LogicalResult fixPipeline(
    DataTransferLegality& legality, DataTransferLegality::PipelineAnalysis& pa,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " alignment_factor=" << pa.alignment_factor
          << " stride_elems=" << pa.stride_elems;

  // Stages to rebuild as a per-element re-stickify. Rebuilt after every other
  // stage of this pipeline: the rewrite reads the source and destination
  // buffers' final types, and a later stage (the store out of this pipeline)
  // may still widen the destination.
  llvm::SmallVector<DataTransferLegality::StageAnalysis*> restickify_stages;

  for (DataTransferLegality::StageAnalysis& sa : pa.stages) {
    if (sa.nested_pipeline != nullptr) {
      // Collapse the tile loop enclosing the nested pipeline to ub=1 so the
      // only column iteration is the element loop inserted below.
      if (mlir::failed(adjustLoopBounds(sa, pa, builder)))
        return mlir::failure();

      // Pass the outer pipeline's stride_elems as an inherited stride so that
      // the nested pipeline's transfers see the full non-contiguous stride and
      // are classified correctly (is_illegal / is_displaced / needs_splat)
      // without any IR types being modified.
      llvm::SmallVector<int64_t, 1> inherited_strides = {pa.stride_elems};
      *sa.nested_pipeline = legality.analyzePipeline(
          sa.nested_pipeline->pipeline, resource_kinds, inherited_strides);
      LDBG(1) << "  nested PipelineAnalysis:\n" << *sa.nested_pipeline;

      // A nested pipeline that loads across rows (its load needs a splat:
      // one element per transfer) re-sticks the data; it is rebuilt as the
      // per-element re-stickify.
      const bool loads_across_rows = llvm::any_of(
          sa.nested_pipeline->stages, [](const auto& nested_stage) {
            return llvm::any_of(nested_stage.transfers,
                                [](const auto& ts) { return ts.needs_splat; });
          });
      if (sa.nested_pipeline->alignment_kind ==
              DataTransferLegality::AlignmentKind::Shrink &&
          loads_across_rows) {
        restickify_stages.push_back(&sa);
        continue;
      }

      insertLoopAroundPipeline(sa.nested_pipeline.get(), pa.stride_elems,
                               builder);
      if (mlir::failed(fixPipeline(legality, *sa.nested_pipeline,
                                   resource_kinds, memory_tree, builder)))
        return mlir::failure();
      continue;
    }

    // Adjust the dimension loop bound for this stage before visiting
    // its transfers.
    if (mlir::failed(adjustLoopBounds(sa, pa, builder))) return mlir::failure();

    for (DataTransferLegality::TransferStep& ts : sa.transfers) {
      if (pa.alignment_kind == DataTransferLegality::AlignmentKind::Widen) {
        // Outer pipeline: widen staging allocs and bulk transfer shapes
        if (ts.is_illegal || ts.is_displaced) {
          widenAlloc(ts, pa, memory_tree, builder);
          if (mlir::failed(rewriteTransferShape(ts, pa, memory_tree, builder)))
            return mlir::failure();
        }
      } else {
        // Nested / inner pipeline: shrink transfers to scalar access with IV
        // indexing
        if (ts.needs_splat) {
          rewriteTransferShrink(ts, pa, memory_tree, "splat", builder);
        } else if (ts.transfer.isSourceFifo() && ts.transfer.isDestMemRef()) {
          rewriteTransferShrink(ts, pa, memory_tree, "", builder);
        }
      }
    }
  }
  for (DataTransferLegality::StageAnalysis* sa : restickify_stages) {
    if (mlir::failed(buildPerElementRestickify(*sa, builder)))
      return mlir::failure();
  }
  return mlir::success();
}

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    llvm::errs() << "[" PASS_NAME "] running on: " << getOperation()->getName()
                 << "\n";
    LDBG(1) << "========= " PASS_NAME " =========";

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* device = device_manager.getOrImportDevice();
    if (!device) {
      getOperation()->emitError(PASS_NAME ": failed to import device spec");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);
    auto& memory_tree =
        device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(
            *device);

    mlir::OpBuilder builder(getOperation()->getContext());

    // Before analysis: the analysis reads each view's dim order.
    llvm::SmallVector<mlir::ktdf::PipelineOp> top_level;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          top_level.push_back(pipeline);
          return mlir::WalkResult::skip();
        });
    for (mlir::ktdf::PipelineOp pipeline : top_level) {
      if (mlir::failed(
              moveStrideOneDimToRowPosition(pipeline, memory_tree, builder))) {
        signalPassFailure();
        return;
      }
    }

    llvm::SmallVector<DataTransferLegality::PipelineAnalysis> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(
              legality_.analyzePipeline(pipeline, resource_kinds));
          return mlir::WalkResult::skip();
        });

    for (auto& pa : pipelines) {
      LDBG(1) << pa;
      if (pa.alignment_factor == 0)
        continue;  // nothing to fix for this pipeline
      if (mlir::failed(fixPipeline(legality_, pa, resource_kinds, memory_tree,
                                   builder))) {
        signalPassFailure();
        return;
      }
    }

    // Every remaining transfer on a global view whose strides do not
    // decrease, including those in pipelines needing no widening, which the
    // loop above skips. Collected first: the rewrite replaces ops.
    llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
    getOperation()->walk(
        [&](mlir::ktdf::DataTransferOp dt) { transfers.push_back(dt); });
    for (mlir::ktdf::DataTransferOp dt : transfers) {
      if (mlir::failed(reorderGlobalViewsOf(dt, memory_tree, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  DataTransferLegality legality_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
