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
// As a separate step, every strided global view is then put in memory order
// (strides decreasing), with the transfers on it reordered to match, since
// downstream layout analysis reads views as row-major.
//
// Ordering: After stage-coarsening + canonicalize. Before double-buffering.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <limits>
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
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
        innermost_loop;  // point loop of p in this stage (set by fixPipeline)
  };

  /// Whether the pass needs to widen transfers to cover a full contiguous
  /// block, or shrink them to a single element (element-addressable memory).
  enum class AlignmentKind { Widen, Shrink };

  /// Root node of a per-pipeline analysis tree. Shared parameters
  /// (alignment_kind, align_dim, alignment_factor) are stored here so all child
  /// nodes can read them directly.
  ///
  /// The strided global view indexes a contiguous buffer through a permutation
  /// of its dims. A transfer moves along the view's last dim q, and is illegal
  /// because q is not the buffer's contiguous dim p (the stride-1 dim). Every
  /// rewrite is the transposition of p and q: the load reads blocks with p
  /// innermost, the LX stage exchanges p and q, the store writes blocks with q
  /// innermost. All other dims are untouched.
  struct PipelineAnalysis {
    mlir::ktdf::PipelineOp pipeline;
    llvm::SmallVector<mlir::memref::AllocOp>
        allocs;  // staging buffer allocs in ktdf.private
    AlignmentKind alignment_kind = AlignmentKind::Widen;
    int64_t align_dim = -1;  // which dim to align, offset from end
    int64_t alignment_factor =
        0;  // factor to multiply the data transfer size by on dim p
    int64_t stride_elems = 0;  // stride of q in the strided view, in elements
    int64_t perm_p = -1;       // view dim with stride 1 (the buffer's lane)
    int64_t perm_q = -1;       // view's innermost dim (the transfer's lane)
    int64_t view_rank = 0;     // rank of the strided global view
    int64_t block_elems = 0;   // extent of p: one aligned block per transfer
    int64_t lx_tile_pos = -1;  // staging-buffer dim indexed by p's point loop
    bool has_error = false;    // analysis emitted an error; the pass must fail
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
          // Widen: find the buffer's contiguous dim p (stride 1) and the
          // transfer's lane q (the view's innermost dim). A contiguous block
          // runs n_p elements along p; each transfer currently covers
          // elems_per_iter of them, so it must be widened by
          // n_p / elems_per_iter.
          if (mlir::failed(
                  analyzePermutation(pa, dt, strided_memref, use_dest)))
            pa.has_error = true;
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
  /// Fills pa's Widen parameters from the illegal transfer `dt`, whose strided
  /// global operand (its destination when `use_dest`, otherwise its source)
  /// has type `strided_memref`.
  mlir::LogicalResult analyzePermutation(PipelineAnalysis& pa,
                                         mlir::ktdf::DataTransferOp dt,
                                         mlir::MemRefType strided_memref,
                                         bool use_dest) {
    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    if (mlir::failed(strided_memref.getStridesAndOffset(strides, offset)))
      return dt->emitError(PASS_NAME ": strided operand has no strided layout");
    llvm::ArrayRef<int64_t> shape = strided_memref.getShape();
    int64_t rank = (int64_t)shape.size();

    // p: the buffer's contiguous dim. Unit dims never move, so their stride is
    // irrelevant and they are not candidates.
    int64_t p = -1;
    for (int64_t k = 0; k < rank; ++k) {
      if (strides[k] != 1 || shape[k] == 1) continue;
      if (p != -1)
        return dt->emitError(PASS_NAME
                             ": strided operand has more than one stride-1 "
                             "dim");
      p = k;
    }
    int64_t q = rank - 1;
    if (p == -1)
      return dt->emitError(PASS_NAME ": strided operand has no stride-1 dim");
    if (p == q)
      return dt->emitError(PASS_NAME
                           ": transfer lane is already the contiguous dim");
    // Exchanging p and q keeps the view's shape only when they have the same
    // extent; the LX transpose also needs a square block.
    if (shape[p] != shape[q])
      return dt->emitError(PASS_NAME ": contiguous dim (extent ")
             << shape[p] << ") and transfer lane (extent " << shape[q]
             << ") must have the same extent to be exchanged";

    auto sizes = use_dest ? dt.getStaticDestSizesArray()
                          : dt.getStaticSourceSizesArray();
    if (!sizes) return dt->emitError(PASS_NAME ": transfer has dynamic sizes");
    int64_t elems_per_iter = std::max<int64_t>(1, (*sizes)[p]);
    if (shape[p] % elems_per_iter != 0)
      return dt->emitError(PASS_NAME
                           ": transfer size along the contiguous dim (")
             << elems_per_iter << ") does not divide its extent (" << shape[p]
             << ")";
    if ((*sizes)[q] != shape[q])
      return dt->emitError(PASS_NAME
                           ": transfer must cover the whole transfer lane");

    auto lx_tile_pos = findStagingTilePos(dt, use_dest, p);
    if (mlir::failed(lx_tile_pos)) return mlir::failure();

    pa.alignment_kind = AlignmentKind::Widen;
    pa.align_dim = rank - 1 - p;
    pa.perm_p = p;
    pa.perm_q = q;
    pa.view_rank = rank;
    pa.block_elems = shape[p];
    pa.stride_elems = strides[q];
    pa.alignment_factor = shape[p] / elems_per_iter;
    pa.lx_tile_pos = *lx_tile_pos;
    return mlir::success();
  }

  /// Returns the staging-buffer dim indexed by the point loop of view dim `p`:
  /// the loop IV behind dt's global index at `p` (directly, or as the point IV
  /// of a ktdf.tiling.linearize_index), located among the indices of dt's
  /// other (staging) operand.
  mlir::FailureOr<int64_t> findStagingTilePos(mlir::ktdf::DataTransferOp dt,
                                              bool global_is_dest, int64_t p) {
    if (!dt.isSourceMemRef() || !dt.isDestMemRef())
      return dt->emitError(PASS_NAME
                           ": illegal transfer must be memref to memref");
    mlir::AffineMap global_map = global_is_dest
                                     ? dt.getDestMapAttr().getValue()
                                     : dt.getSourceMapAttr().getValue();
    mlir::OperandRange global_indices =
        global_is_dest ? dt.getDestIndices() : dt.getSourceIndices();
    mlir::AffineMap staging_map = global_is_dest
                                      ? dt.getSourceMapAttr().getValue()
                                      : dt.getDestMapAttr().getValue();
    mlir::OperandRange staging_indices =
        global_is_dest ? dt.getSourceIndices() : dt.getDestIndices();

    auto global_dim =
        mlir::dyn_cast<mlir::AffineDimExpr>(global_map.getResult(p));
    if (!global_dim)
      return dt->emitError(PASS_NAME
                           ": global index along the contiguous dim is not a "
                           "plain index");
    mlir::Value iv = global_indices[global_dim.getPosition()];
    if (auto linearize = iv.getDefiningOp<mlir::ktdf::TilingLinearizeIndexOp>())
      iv = linearize.getIvs().back();
    if (!mlir::scf::getForInductionVarOwner(iv))
      return dt->emitError(PASS_NAME
                           ": global index along the contiguous dim is not "
                           "driven by a loop");

    for (auto [pos, expr] : llvm::enumerate(staging_map.getResults())) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (dim && staging_indices[dim.getPosition()] == iv) return (int64_t)pos;
    }
    return dt->emitError(PASS_NAME
                         ": the contiguous dim's loop does not index the "
                         "staging buffer");
  }

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
      return sa;
    }

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
  os << indent << "  perm_p=" << pa.perm_p << " perm_q=" << pa.perm_q
     << " view_rank=" << pa.view_rank << " block_elems=" << pa.block_elems
     << " lx_tile_pos=" << pa.lx_tile_pos << "\n";
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

/// Widens any ct_local alloc backing a source or destination of `ts` along
/// view dim p by pa.alignment_factor. Non-ct_local sides are skipped.
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

    // The staging buffer is the transfer's block shape (one dim per view dim,
    // trailing) behind its tile dims. Widen the block along p, and shrink p's
    // tile dim (pa.lx_tile_pos) by the same factor: each slot now holds
    // alignment_factor of p's points, and adjustLoopBounds makes p's tile a
    // multiple of alignment_factor.
    int64_t rank = orig_type.getRank();
    int64_t alloc_dim = rank - pa.view_rank + pa.perm_p;
    int64_t tile_dim = pa.lx_tile_pos;
    if (alloc_dim < 0 || alloc_dim >= rank || tile_dim < 0 ||
        tile_dim >= alloc_dim) {
      ts.transfer->emitError("widenAlloc: staging buffer of rank ")
          << rank << " has no block dim for p (" << alloc_dim
          << ") behind its tile dim (" << tile_dim << ")";
      return;
    }

    llvm::SmallVector<mlir::Value> new_dynamic_sizes(alloc.getDynamicSizes());
    if (orig_type.isDynamicDim(tile_dim)) {
      builder.setInsertionPoint(alloc);
      mlir::Value c_factor = mlir::arith::ConstantIndexOp::create(
                                 builder, alloc.getLoc(), alignment_factor)
                                 .getResult();
      mlir::Value& tile_size =
          new_dynamic_sizes[orig_type.getDynamicDimIndex(tile_dim)];
      tile_size = mlir::arith::DivUIOp::create(builder, alloc.getLoc(),
                                               tile_size, c_factor)
                      .getResult();
    } else {
      if (new_shape[tile_dim] % alignment_factor != 0) {
        ts.transfer->emitError("widenAlloc: tile dimension (")
            << new_shape[tile_dim] << ") is not divisible by alignment factor "
            << alignment_factor;
        return;
      }
      new_shape[tile_dim] = new_shape[tile_dim] / alignment_factor;
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

/// Returns the point loop of view dim p in `stage`: the scf.for whose IV
/// indexes a staging buffer at pa.lx_tile_pos. Covers both leaf stages and a
/// stage wrapping a nested pipeline, whose element transfers index the same
/// staging buffers with the stage's loop IVs.
static mlir::scf::ForOp findPointLoop(
    mlir::ktdf::StageOp stage, const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  mlir::scf::ForOp point_loop;
  stage->walk([&](mlir::ktdf::DataTransferOp dt) {
    auto check = [&](mlir::Value memref_val, mlir::AffineMap map,
                     mlir::OperandRange indices) {
      if (point_loop || !isPerCoreScratchpad(memref_val, memory_tree)) return;
      if (pa.lx_tile_pos >= (int64_t)map.getNumResults()) return;
      auto dim =
          mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(pa.lx_tile_pos));
      if (!dim) return;
      point_loop =
          mlir::scf::getForInductionVarOwner(indices[dim.getPosition()]);
    };
    if (dt.isSourceMemRef())
      check(dt.getSource(), dt.getSourceMapAttr().getValue(),
            dt.getSourceIndices());
    if (dt.isDestMemRef())
      check(dt.getDestination(), dt.getDestMapAttr().getValue(),
            dt.getDestIndices());
  });
  return point_loop;
}

/// Makes the tile behind `tile_size` hold whole aligned blocks: a
/// ktdf.tiling.reserve_size is constrained to a multiple of alignment_factor
/// before tile size selection runs; an already chosen constant tile size is
/// checked instead.
static mlir::LogicalResult constrainTileToAlignedBlocks(
    mlir::Value tile_size, int64_t alignment_factor, mlir::Operation* anchor,
    mlir::OpBuilder& builder) {
  mlir::Operation* def = tile_size.getDefiningOp();
  if (auto reserve =
          mlir::dyn_cast_or_null<mlir::ktdf::TilingReserveSizeOp>(def)) {
    int64_t divisibility =
        std::lcm(reserve.getDivisibility().getSExtValue(), alignment_factor);
    int64_t min_value =
        std::max(reserve.getMinValue().getSExtValue(), alignment_factor);
    reserve.setDivisibilityAttr(builder.getIndexAttr(divisibility));
    reserve.setMinValueAttr(builder.getIndexAttr(min_value));
    LDBG(1) << "  constrainTileToAlignedBlocks: " << reserve;
    return mlir::success();
  }
  if (auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(def)) {
    if (cst.value() % alignment_factor == 0) return mlir::success();
    return anchor->emitError(PASS_NAME ": tile size (")
           << cst.value() << ") is not a multiple of alignment factor "
           << alignment_factor;
  }
  return anchor->emitError(
             PASS_NAME
             ": tile size is neither a ktdf.tiling.reserve_size nor a constant "
             "— cannot constrain it to alignment_factor=")
         << alignment_factor;
}

/// Checks that `total_size`, a dimension's full trip count, is a constant
/// divisible by alignment_factor. `what` names it in the error.
static mlir::LogicalResult checkTotalDivisible(mlir::Value total_size,
                                               int64_t alignment_factor,
                                               mlir::Operation* anchor,
                                               llvm::StringRef what) {
  auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
      total_size.getDefiningOp());
  if (!cst) {
    return anchor->emitError(PASS_NAME ": ")
           << what
           << " is not a constant — cannot verify dimension size against "
              "alignment_factor="
           << alignment_factor;
  }
  if (alignment_factor <= 0 || cst.value() % alignment_factor != 0) {
    return anchor->emitError(PASS_NAME ": dimension total size (")
           << cst.value() << ") is not divisible by alignment factor "
           << alignment_factor
           << "; tiling and alignment constraints are inconsistent";
  }
  return mlir::success();
}

/// Point loop of a tiled dimension: its bound `derive` is the extent of one
/// tile, so the tile is constrained to whole aligned blocks, the per-tile trip
/// count is divided by alignment_factor, and the loop's stride in every
/// ktdf.tiling.linearize_index is scaled by alignment_factor so that each
/// iteration advances one aligned block in the original iteration space.
static mlir::LogicalResult adjustTiledLoopBound(
    mlir::scf::ForOp loop, mlir::ktdf::TilingDeriveSizeOp derive,
    int64_t alignment_factor, mlir::OpBuilder& builder) {
  if (mlir::failed(checkTotalDivisible(derive.getTotalSize(), alignment_factor,
                                       loop,
                                       "ktdf.tiling.derive_size total_size")))
    return mlir::failure();

  // The innermost tile size is the extent this loop walks.
  if (mlir::failed(constrainTileToAlignedBlocks(
          derive.getTileSizes().back(), alignment_factor, loop, builder)))
    return mlir::failure();

  // Both the full tile and the epilogue remainder are multiples of
  // alignment_factor (the tile is constrained, the total checked above), so the
  // division is exact.
  builder.setInsertionPoint(loop);
  mlir::Value c_factor = mlir::arith::ConstantIndexOp::create(
                             builder, loop.getLoc(), alignment_factor)
                             .getResult();
  loop.setUpperBound(
      mlir::arith::DivUIOp::create(builder, loop.getLoc(), derive, c_factor)
          .getResult());

  mlir::Value iv = loop.getInductionVar();
  for (mlir::Operation* user : llvm::to_vector(iv.getUsers())) {
    auto linearize = mlir::dyn_cast<mlir::ktdf::TilingLinearizeIndexOp>(user);
    if (!linearize) continue;
    for (auto [idx, linearize_iv] : llvm::enumerate(linearize.getIvs())) {
      if (linearize_iv != iv) continue;
      builder.setInsertionPoint(linearize);
      mlir::Value stride = linearize.getStrides()[idx];
      mlir::Value scaled;
      if (auto stride_cst =
              mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(
                  stride.getDefiningOp())) {
        scaled = mlir::arith::ConstantIndexOp::create(
                     builder, linearize.getLoc(),
                     stride_cst.value() * alignment_factor)
                     .getResult();
      } else {
        scaled = mlir::arith::MulIOp::create(builder, linearize.getLoc(),
                                             stride, c_factor)
                     .getResult();
      }
      linearize.getStridesMutable()[idx].assign(scaled);
    }
  }

  LDBG(1) << "  adjustTiledLoopBound: per-tile trip count / alignment_factor="
          << alignment_factor << " at " << loop.getLoc();
  return mlir::success();
}

/// Untiled loop whose bound is the constant full trip count: it only needs
/// total / alignment_factor iterations. Its IV indexes the global view
/// directly, unscaled, so this is only correct when the dimension is a single
/// aligned block (one iteration, IV 0).
static mlir::LogicalResult adjustConstantLoopBound(mlir::scf::ForOp loop,
                                                   int64_t alignment_factor,
                                                   mlir::OpBuilder& builder) {
  if (mlir::failed(checkTotalDivisible(loop.getUpperBound(), alignment_factor,
                                       loop, "point loop upper bound")))
    return mlir::failure();

  int64_t total_val = mlir::cast<mlir::arith::ConstantIndexOp>(
                          loop.getUpperBound().getDefiningOp())
                          .value();
  if (total_val != alignment_factor) {
    return loop->emitError(PASS_NAME ": untiled dimension (")
           << total_val << ") spans more than one aligned block of "
           << alignment_factor << "; only tiled dimensions are supported";
  }
  int64_t new_ub = total_val / alignment_factor;
  builder.setInsertionPoint(loop);
  mlir::Value new_ub_val =
      mlir::arith::ConstantIndexOp::create(builder, loop.getLoc(), new_ub)
          .getResult();
  loop.setUpperBound(new_ub_val);

  LDBG(1) << "  adjustConstantLoopBound: total_size=" << total_val
          << " / alignment_factor=" << alignment_factor
          << " → new ub=" << new_ub << " at " << loop.getLoc();
  return mlir::success();
}

/// Each widened transfer covers alignment_factor times as much data per
/// iteration, so sa.innermost_loop needs alignment_factor times fewer
/// iterations. Its upper bound is normally a ktdf.tiling.derive_size result
/// (produced by StageCoarseningPass); when the IR was not tiled it may instead
/// be a bare arith.constant.
static mlir::LogicalResult adjustLoopBounds(
    const DataTransferLegality::StageAnalysis& sa,
    const DataTransferLegality::PipelineAnalysis& pa,
    mlir::OpBuilder& builder) {
  if (!sa.innermost_loop) return mlir::success();

  mlir::scf::ForOp loop = sa.innermost_loop;
  if (auto derive = mlir::dyn_cast_or_null<mlir::ktdf::TilingDeriveSizeOp>(
          loop.getUpperBound().getDefiningOp()))
    return adjustTiledLoopBound(loop, derive, pa.alignment_factor, builder);
  return adjustConstantLoopBound(loop, pa.alignment_factor, builder);
}

/// Returns the ktdp.construct_memory_view behind a global (non-ct_local)
/// memref operand, walking up through view-like casts; null if there is none.
static mlir::ktdp::ConstructMemoryViewOp findConstructMemoryView(
    mlir::Value val, const scheduler::arch_view::MemoryTree& memory_tree) {
  if (!mlir::isa<mlir::MemRefType>(val.getType())) return {};
  if (isPerCoreScratchpad(val, memory_tree)) return {};
  mlir::Value cursor = val;
  while (auto def_op = cursor.getDefiningOp()) {
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

/// Returns the order listing a view's dims in memory order: unit dims first
/// (their index is always 0, so their position is free), then the others by
/// decreasing stride. order[k] is the original dim placed at position k.
static llvm::SmallVector<int64_t> memoryOrder(llvm::ArrayRef<int64_t> shape,
                                              llvm::ArrayRef<int64_t> strides) {
  auto order = llvm::to_vector(llvm::seq<int64_t>(0, shape.size()));
  llvm::stable_sort(order, [&](int64_t a, int64_t b) {
    bool a_unit = shape[a] == 1;
    bool b_unit = shape[b] == 1;
    if (a_unit != b_unit) return a_unit;
    if (a_unit) return false;
    return strides[a] > strides[b];
  });
  return order;
}

/// Rewrites `construct` to list its dims in memory order (see memoryOrder),
/// and returns that order. Reordering a view's dims, each keeping its own size
/// and stride, only renames which index slot addresses which buffer dim: the
/// view covers the same elements at the same addresses. Memory order is the
/// one reordering that both puts the contiguous dim p innermost (the hardware
/// moves contiguous sticks) and makes the strides decrease (downstream layout
/// analysis reads views as row-major).
///
/// The sizes, strides, coordinate_set and result type are reordered together,
/// and the new type is propagated through the view-like casts that consume
/// the view. Any other consumer except ktdf.data_transfer (whose maps the
/// caller reorders) is an error.
static mlir::FailureOr<llvm::SmallVector<int64_t>> permuteViewToMemoryOrder(
    mlir::ktdp::ConstructMemoryViewOp construct, mlir::Operation* anchor) {
  mlir::MLIRContext* ctx = construct->getContext();
  auto orig_mrt = mlir::cast<mlir::MemRefType>(construct.getResult().getType());
  llvm::ArrayRef<int64_t> shape = construct.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = construct.getStaticStrides();
  int64_t rank = (int64_t)shape.size();
  if (llvm::any_of(shape, mlir::ShapedType::isDynamic) ||
      llvm::any_of(strides, mlir::ShapedType::isDynamic))
    return anchor->emitError(PASS_NAME
                             ": strided view must have static sizes and "
                             "strides to be reordered");

  llvm::SmallVector<int64_t> order = memoryOrder(shape, strides);
  llvm::SmallVector<int64_t> new_shape, new_strides;
  for (int64_t dim : order) {
    new_shape.push_back(shape[dim]);
    new_strides.push_back(strides[dim]);
  }
  // A unit dim's stride never contributes to an address; raise it where needed
  // so the strides never increase going inwards.
  for (int64_t k = rank - 2; k >= 0; --k)
    if (new_shape[k] == 1)
      new_strides[k] = std::max(new_strides[k], new_strides[k + 1]);

  // Old dim `order[k]` is now dim k.
  mlir::IntegerSet set = construct.getCoordinateSet().getValue();
  llvm::SmallVector<mlir::AffineExpr> dim_replacements(rank);
  for (int64_t k = 0; k < rank; ++k)
    dim_replacements[order[k]] = mlir::getAffineDimExpr(k, ctx);
  llvm::SmallVector<mlir::AffineExpr> constraints;
  for (mlir::AffineExpr expr : set.getConstraints())
    constraints.push_back(expr.replaceDims(dim_replacements));
  mlir::IntegerSet new_set = mlir::IntegerSet::get(
      rank, set.getNumSymbols(), constraints, set.getEqFlags());

  LDBG(1) << "  permuteViewToMemoryOrder: " << construct->getLoc();
  construct.setStaticSizesAttr(mlir::DenseI64ArrayAttr::get(ctx, new_shape));
  construct.setStaticStridesAttr(
      mlir::DenseI64ArrayAttr::get(ctx, new_strides));
  construct.setCoordinateSetAttr(mlir::IntegerSetAttr::get(new_set));
  auto new_layout = mlir::StridedLayoutAttr::get(
      ctx, /*offset=*/mlir::ShapedType::kDynamic, new_strides);
  construct.getResult().setType(
      mlir::MemRefType::get(new_shape, orig_mrt.getElementType(), new_layout,
                            orig_mrt.getMemorySpace()));

  // Propagate the new shape and layout through the view-like casts.
  llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    auto src_type = mlir::cast<mlir::MemRefType>(v.getType());
    for (mlir::Operation* user : v.getUsers()) {
      if (mlir::isa<mlir::ktdf::DataTransferOp>(user)) continue;
      if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp>(
              user))
        return user->emitError(PASS_NAME
                               ": cannot reorder a strided view consumed by "
                               "this op");
      mlir::Value result = user->getResult(0);
      auto res_type = mlir::cast<mlir::MemRefType>(result.getType());
      result.setType(mlir::MemRefType::get(
          src_type.getShape(), res_type.getElementType(), src_type.getLayout(),
          res_type.getMemorySpace()));
      worklist.push_back(result);
    }
  }
  return order;
}

/// Reorders one memref side of a transfer, its `sizes` and the results of its
/// index `map`, by a view's memory `order` (see permuteViewToMemoryOrder).
static void reorderTransferSide(llvm::ArrayRef<int64_t> order,
                                llvm::SmallVector<mlir::OpFoldResult>& sizes,
                                mlir::AffineMap& map) {
  llvm::SmallVector<mlir::OpFoldResult> new_sizes;
  llvm::SmallVector<mlir::AffineExpr> results;
  for (int64_t dim : order) {
    new_sizes.push_back(sizes[dim]);
    results.push_back(map.getResult(dim));
  }
  sizes = new_sizes;
  map = mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                             map.getContext());
}

/// Replaces `op` with a ktdf.data_transfer on the same operands and indices
/// but the given index maps and sizes, keeping its discardable attributes.
static mlir::ktdf::DataTransferOp replaceTransfer(
    mlir::ktdf::DataTransferOp op, mlir::AffineMap src_map,
    llvm::ArrayRef<mlir::OpFoldResult> src_sizes, mlir::AffineMap dst_map,
    llvm::ArrayRef<mlir::OpFoldResult> dst_sizes, mlir::OpBuilder& builder) {
  builder.setInsertionPoint(op);
  auto new_op = mlir::ktdf::DataTransferOp::create(
      builder, op.getLoc(), op.getSource(), src_map, op.getSourceIndices(),
      src_sizes, op.getDestination(), dst_map, op.getDestIndices(), dst_sizes);
  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    new_op->setDiscardableAttr(attr.getName(), attr.getValue());
  op.erase();
  return new_op;
}

/// Replaces ts.transfer with a new DataTransferOp widened along view dim p by
/// pa.alignment_factor.
///
/// A side on the strided global view is rewritten through the view's memory
/// order: the view is reordered once (recorded in `view_orders`), the
/// transfer's index map results and sizes are reordered with it, and the size
/// of p, now the innermost dim, is widened. Every other memref side (staging
/// buffers, row-major views) is widened at p in place.
///
/// The staging block needs no reordering: in a block only p and q span more
/// than one element, and memory order puts q before p, so the block lands in
/// the staging buffer with p and q exchanged — which the element-level
/// transpose undoes.
static mlir::LogicalResult rewriteTransferShape(
    const DataTransferLegality::TransferStep& ts,
    const DataTransferLegality::PipelineAnalysis& pa,
    const scheduler::arch_view::MemoryTree& memory_tree,
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>>& view_orders,
    mlir::OpBuilder& builder) {
  mlir::ktdf::DataTransferOp op = ts.transfer;
  mlir::MLIRContext* ctx = op->getContext();

  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  mlir::AffineMap src_map = op.isSourceMemRef()
                                ? op.getSourceMapAttr().getValue()
                                : mlir::AffineMap{};
  mlir::AffineMap dst_map =
      op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};

  auto get_scaled = [&](mlir::OpFoldResult ofr) -> mlir::OpFoldResult {
    int64_t existing =
        mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(ofr))
            .getInt();
    return mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                  existing * pa.alignment_factor);
  };

  auto rewrite_side = [&](mlir::Value val,
                          llvm::SmallVector<mlir::OpFoldResult>& sizes,
                          mlir::AffineMap& map) -> mlir::LogicalResult {
    if (!mlir::isa<mlir::MemRefType>(val.getType())) return mlir::success();
    int64_t rank = (int64_t)sizes.size();

    // The strided view is the one whose own innermost stride is not 1, or
    // that an earlier transfer already reordered.
    auto construct = findConstructMemoryView(val, memory_tree);
    const llvm::SmallVector<int64_t>* order = nullptr;
    if (construct) {
      auto it = view_orders.find(construct.getOperation());
      if (it == view_orders.end() && construct.getStaticStrides().back() != 1) {
        auto new_order = permuteViewToMemoryOrder(construct, op);
        if (mlir::failed(new_order)) return mlir::failure();
        it =
            view_orders.try_emplace(construct.getOperation(), *new_order).first;
      }
      if (it != view_orders.end()) order = &it->second;
    }

    if (!order) {
      // View dims are the trailing dims of every memref side.
      int64_t pos_p = rank - pa.view_rank + pa.perm_p;
      sizes[pos_p] = get_scaled(sizes[pos_p]);
      return mlir::success();
    }
    if (rank != (int64_t)order->size() || (*order)[rank - 1] != pa.perm_p)
      return op->emitError(PASS_NAME
                           ": memory order of the strided view does not end "
                           "with its contiguous dim");
    reorderTransferSide(*order, sizes, map);
    sizes[rank - 1] = get_scaled(sizes[rank - 1]);
    return mlir::success();
  };

  if (mlir::failed(rewrite_side(op.getSource(), new_src_sizes, src_map)) ||
      mlir::failed(rewrite_side(op.getDestination(), new_dst_sizes, dst_map)))
    return mlir::failure();

  auto new_op = replaceTransfer(op, src_map, new_src_sizes, dst_map,
                                new_dst_sizes, builder);
  LDBG(1) << "  rewriteTransferShape: widened dim " << pa.perm_p
          << " by factor " << pa.alignment_factor << ": " << new_op;
  return mlir::success();
}

/// Returns true if `construct` is a global view whose static strides do not
/// already list its dims in memory order, and whose only consumers are
/// view-like casts and transfers (so it can be reordered).
static bool needsMemoryOrder(
    mlir::ktdp::ConstructMemoryViewOp construct,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  if (isPerCoreScratchpad(construct.getResult(), memory_tree)) return false;
  llvm::ArrayRef<int64_t> shape = construct.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = construct.getStaticStrides();
  if (llvm::any_of(shape, mlir::ShapedType::isDynamic) ||
      llvm::any_of(strides, mlir::ShapedType::isDynamic))
    return false;
  // In memory order when the strides never increase going inwards; unit dims
  // never move, so their stride is irrelevant.
  int64_t prev = std::numeric_limits<int64_t>::max();
  bool ordered = true;
  for (auto [size, stride] : llvm::zip(shape, strides)) {
    if (size == 1) continue;
    if (stride > prev) ordered = false;
    prev = stride;
  }
  if (ordered) return false;

  llvm::SmallVector<mlir::Value> worklist = {construct.getResult()};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    for (mlir::Operation* user : v.getUsers()) {
      if (mlir::isa<mlir::ktdf::DataTransferOp>(user)) continue;
      if (!mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp>(
              user))
        return false;
      worklist.push_back(user->getResult(0));
    }
  }
  return true;
}

/// Reorders into memory order every strided global view that fixPipeline did
/// not already reorder, together with the transfers on it.
///
/// Downstream layout analysis reads a view's strides as decreasing, so a view
/// whose outer dims are out of memory order cannot be lowered even when every
/// transfer on it is legal — e.g. a transpose that only exchanges outer dims
/// and keeps the contiguous dim innermost. Reordering only renames which index
/// slot addresses which buffer dim (see permuteViewToMemoryOrder), so the
/// transfers move the same elements; nothing is widened.
static mlir::LogicalResult reorderViewsToMemoryOrder(
    mlir::Operation* root, const scheduler::arch_view::MemoryTree& memory_tree,
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>>& view_orders,
    mlir::OpBuilder& builder) {
  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>> new_orders;
  mlir::WalkResult walk =
      root->walk([&](mlir::ktdp::ConstructMemoryViewOp construct) {
        if (view_orders.contains(construct.getOperation()) ||
            !needsMemoryOrder(construct, memory_tree))
          return mlir::WalkResult::advance();
        auto order = permuteViewToMemoryOrder(construct, construct);
        if (mlir::failed(order)) return mlir::WalkResult::interrupt();
        new_orders.try_emplace(construct.getOperation(), *order);
        return mlir::WalkResult::advance();
      });
  if (walk.wasInterrupted()) return mlir::failure();
  if (new_orders.empty()) return mlir::success();

  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  root->walk([&](mlir::ktdf::DataTransferOp op) { transfers.push_back(op); });
  for (mlir::ktdf::DataTransferOp op : transfers) {
    auto order_of = [&](mlir::Value val) -> const llvm::SmallVector<int64_t>* {
      auto construct = findConstructMemoryView(val, memory_tree);
      if (!construct) return nullptr;
      auto it = new_orders.find(construct.getOperation());
      return it == new_orders.end() ? nullptr : &it->second;
    };
    auto src_order = op.isSourceMemRef() ? order_of(op.getSource()) : nullptr;
    auto dst_order =
        op.isDestMemRef() ? order_of(op.getDestination()) : nullptr;
    if (!src_order && !dst_order) continue;

    auto src_sizes = op.getMixedSourceSizes();
    auto dst_sizes = op.getMixedDestSizes();
    mlir::AffineMap src_map = op.isSourceMemRef()
                                  ? op.getSourceMapAttr().getValue()
                                  : mlir::AffineMap{};
    mlir::AffineMap dst_map =
        op.isDestMemRef() ? op.getDestMapAttr().getValue() : mlir::AffineMap{};
    if (src_order) reorderTransferSide(*src_order, src_sizes, src_map);
    if (dst_order) reorderTransferSide(*dst_order, dst_sizes, dst_map);
    auto new_op =
        replaceTransfer(op, src_map, src_sizes, dst_map, dst_sizes, builder);
    LDBG(1) << "  reorderViewsToMemoryOrder: " << new_op;
  }

  for (auto& [construct, order] : new_orders)
    view_orders.try_emplace(construct, order);
  return mlir::success();
}

/// Replaces ts.transfer with a new DataTransferOp whose memref side is sized
/// [1,1,1,1] and whose FIFO side keeps its slot size [E], with transfer_mode
/// set to `mode`:
///   - memref→FIFO, "splat": one element is read and broadcast across the
///     slot.
///   - FIFO→memref, "lane0": the slot carries a full vector and its lane 0 is
///     written as the one element at the destination indices.
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

  // Collapse the memref side to [1,1,1,1], one element, and keep the FIFO side
  // at its slot size [E].
  auto new_src_sizes = op.getMixedSourceSizes();
  auto new_dst_sizes = op.getMixedDestSizes();
  if (op.isSourceMemRef())
    for (auto& s : new_src_sizes) s = one;
  if (op.isDestMemRef())
    for (auto& s : new_dst_sizes) s = one;

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

  // For ct_local memref operands, offset the staging buffer's p and q dims by
  // the element IVs. The other dims keep the transfer's own indices: they
  // select which aligned block of the staging buffer this element belongs to.
  // The load (source) buffer holds the block with p and q exchanged (the view
  // was read through the transposition): index [p: %row, q: %col].
  // The store (destination) buffer holds it in view order: index
  // [p: %col, q: %row] — the exchange of p and q is undone here.
  auto apply_row_col = [&](mlir::Value memref_val, mlir::AffineMap& map,
                           llvm::SmallVector<mlir::Value>& indices,
                           bool transpose) {
    if (!row_iv || !col_iv) return;
    if (!isPerCoreScratchpad(memref_val, memory_tree)) return;
    auto mrt = mlir::cast<mlir::MemRefType>(memref_val.getType());

    // View dims are the trailing dims of the staging buffer.
    int64_t rank = (int64_t)mrt.getRank();
    int64_t pos_p = rank - pa.view_rank + pa.perm_p;
    int64_t pos_q = rank - pa.view_rank + pa.perm_q;
    if (pos_p < 0 || pos_q >= rank) return;

    // The element IVs become two new dims after the map's existing ones.
    // For load: (row, col) at (p, q).
    // For store (transposed): (col, row) at (p, q) i.e. swap the IVs.
    mlir::Value first_iv = transpose ? col_iv : row_iv;
    mlir::Value second_iv = transpose ? row_iv : col_iv;

    unsigned num_dims = map.getNumDims();
    llvm::SmallVector<mlir::AffineExpr> results(map.getResults());
    results[pos_p] = results[pos_p] + mlir::getAffineDimExpr(num_dims, ctx);
    results[pos_q] = results[pos_q] + mlir::getAffineDimExpr(num_dims + 1, ctx);
    map = mlir::AffineMap::get(num_dims + 2, map.getNumSymbols(), results, ctx);
    // Map operands are the dims followed by the symbols.
    indices.insert(indices.begin() + num_dims, {first_iv, second_iv});
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

/// Applies corrective rewrites to the PipelineAnalysis tree: inserts a loop
/// around nested pipelines, then for each leaf stage adjusts the loop bound
/// and rewrites every transfer shape (illegal/displaced → widen; memref→FIFO
/// → splat shrink; FIFO→memref → lane0 shrink).
static mlir::LogicalResult fixPipeline(
    DataTransferLegality& legality, DataTransferLegality::PipelineAnalysis& pa,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    const scheduler::arch_view::MemoryTree& memory_tree,
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>>& view_orders,
    mlir::OpBuilder& builder) {
  LDBG(1) << "  fixPipeline: pipeline at " << pa.pipeline.getLoc()
          << " alignment_factor=" << pa.alignment_factor
          << " stride_elems=" << pa.stride_elems;
  if (pa.has_error) return mlir::failure();
  if (pa.alignment_factor == 0) return mlir::success();  // nothing to fix

  for (DataTransferLegality::StageAnalysis& sa : pa.stages) {
    // In the widening pipeline every stage walks p with one loop: the one
    // indexing the staging buffers at p's tile dim. That loop, not the
    // stage's innermost one, is the one whose trip count shrinks.
    if (pa.alignment_kind == DataTransferLegality::AlignmentKind::Widen) {
      sa.innermost_loop = findPointLoop(sa.stage, pa, memory_tree);
      if (!sa.innermost_loop)
        return sa.stage->emitError(PASS_NAME
                                   ": no loop in this stage indexes the "
                                   "staging buffer along the contiguous dim");
    }

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
      // The element transfers address the same staging buffers, so they need
      // the outer pipeline's p/q geometry.
      sa.nested_pipeline->perm_p = pa.perm_p;
      sa.nested_pipeline->perm_q = pa.perm_q;
      sa.nested_pipeline->view_rank = pa.view_rank;
      sa.nested_pipeline->block_elems = pa.block_elems;
      sa.nested_pipeline->lx_tile_pos = pa.lx_tile_pos;
      LDBG(1) << "  nested PipelineAnalysis:\n" << *sa.nested_pipeline;

      // One element loop per block dim: p and q both span block_elems (the
      // analysis requires n_p == n_q).
      insertLoopAroundPipeline(sa.nested_pipeline.get(), pa.block_elems,
                               builder);
      if (mlir::failed(fixPipeline(legality, *sa.nested_pipeline,
                                   resource_kinds, memory_tree, view_orders,
                                   builder)))
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
          if (mlir::failed(rewriteTransferShape(ts, pa, memory_tree,
                                                view_orders, builder)))
            return mlir::failure();
        }
      } else {
        // Nested / inner pipeline: shrink transfers to scalar access with IV
        // indexing
        if (ts.needs_splat) {
          rewriteTransferShrink(ts, pa, memory_tree, "splat", builder);
        } else if (ts.transfer.isSourceFifo() && ts.transfer.isDestMemRef()) {
          rewriteTransferShrink(ts, pa, memory_tree, "lane0", builder);
        }
      }
    }
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

    llvm::SmallVector<DataTransferLegality::PipelineAnalysis> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(
              legality_.analyzePipeline(pipeline, resource_kinds));
          return mlir::WalkResult::skip();
        });

    mlir::OpBuilder builder(getOperation()->getContext());
    // Strided views already reordered into memory order, with their order; a
    // view shared by several transfers must be reordered only once.
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<int64_t>> view_orders;
    for (auto& pa : pipelines) {
      LDBG(1) << pa;
      if (pa.has_error) {
        signalPassFailure();
        return;
      }
      if (pa.alignment_factor == 0)
        continue;  // nothing to fix for this pipeline
      if (mlir::failed(fixPipeline(legality_, pa, resource_kinds, memory_tree,
                                   view_orders, builder))) {
        signalPassFailure();
        return;
      }
    }

    // Separately from alignment: every remaining strided global view must
    // also reach the downstream layout analysis in memory order.
    if (mlir::failed(reorderViewsToMemoryOrder(getOperation(), memory_tree,
                                               view_orders, builder)))
      signalPassFailure();
  }

 private:
  DataTransferLegality legality_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
