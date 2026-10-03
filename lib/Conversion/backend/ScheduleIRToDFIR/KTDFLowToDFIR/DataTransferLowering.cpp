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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Gets the value of the `dataflow_scheduler.throttle` attribute, if any.
[[nodiscard]] auto getThrottle(mlir::Operation* op) -> std::optional<int64_t> {
  if (const auto attr = llvm::dyn_cast_if_present<mlir::ktdf_arch::I64Attr>(
          op->getDiscardableAttr(kThrottleAttrName));
      attr) {
    return attr.getValue();
  }

  return std::nullopt;
}

/// Create a vectorchain.shuffle that broadcasts src_vec (vector<src_elements x
/// T>) to vector<dst_elements x T> using indices [0..src_elements-1] repeated
/// (dst_elements / src_elements) times.
mlir::Value insertSplatShuffle(mlir::PatternRewriter& rewriter,
                               mlir::Location loc, mlir::Value src_vec,
                               int64_t src_elements, int64_t dst_elements) {
  auto src_vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  auto elem_type = src_vec_type.getElementType();

  // Build indices [0, 1, ..., src_elements-1]
  llvm::SmallVector<mlir::Attribute> index_attrs;
  for (int64_t i = 0; i < src_elements; ++i) {
    index_attrs.push_back(rewriter.getIntegerAttr(rewriter.getI32Type(), i));
  }
  auto indices_attr = rewriter.getArrayAttr(index_attrs);

  int32_t repetition = static_cast<int32_t>(dst_elements / src_elements);
  auto result_type = mlir::VectorType::get({dst_elements}, elem_type);

  return mlir::vectorchain::ShuffleOp::create(
             rewriter, loc, result_type, src_vec,
             /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
             /*mask=*/nullptr,
             /*dbgName=*/nullptr, indices_attr,
             rewriter.getI32IntegerAttr(repetition))
      .getOutput();
}

/// Create a vectorchain.shuffle that takes the leading `num_elements` lanes of
/// src_vec, giving vector<num_elements x T> with indices [0..num_elements-1]
/// and repetition 1.
mlir::Value insertLeadingLanesShuffle(mlir::PatternRewriter& rewriter,
                                      mlir::Location loc, mlir::Value src_vec,
                                      int64_t num_elements) {
  auto src_vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  auto elem_type = src_vec_type.getElementType();

  llvm::SmallVector<mlir::Attribute> index_attrs;
  for (int64_t i = 0; i < num_elements; ++i) {
    index_attrs.push_back(rewriter.getIntegerAttr(rewriter.getI32Type(), i));
  }
  auto indices_attr = rewriter.getArrayAttr(index_attrs);

  auto result_type = mlir::VectorType::get({num_elements}, elem_type);

  return mlir::vectorchain::ShuffleOp::create(
             rewriter, loc, result_type, src_vec,
             /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
             /*mask=*/nullptr,
             /*dbgName=*/nullptr, indices_attr, rewriter.getI32IntegerAttr(1))
      .getOutput();
}

/// One time dimension of one side of a transfer: which memref dimension it
/// advances, and by how many indices of that dimension per step.
struct TransferTimeStep {
  unsigned memref_dim;
  int64_t index_step;
};

/// How one side of a transfer traverses the AGEN time axis: the extent of each
/// time dimension, slowest-varying first, and what each dimension advances.
/// `extents` becomes `time_set`; `offsets()` becomes the results of that
/// side's `*_time_addr_map`.
struct TransferTimeDims {
  llvm::SmallVector<int64_t> extents;
  llvm::SmallVector<TransferTimeStep> steps;  // parallel to `extents`
  size_t rank = 0;

  /// A traversal of a memref of `rank` dimensions that walks nothing.
  explicit TransferTimeDims(size_t rank) : rank(rank) {}

  /// The offset added to each memref index at time step (d0, ..., dn-1); zero
  /// at every index a time dimension does not advance. Time dimensions are
  /// numbered in the order they were added, which is also why an identity
  /// `time_order` is correct: d0 is the slowest-varying.
  llvm::SmallVector<mlir::AffineExpr> offsets(
      mlir::MLIRContext* context) const {
    llvm::SmallVector<mlir::AffineExpr> result(
        rank, mlir::getAffineConstantExpr(0, context));
    for (auto [time_dim, step] : llvm::enumerate(steps)) {
      result[step.memref_dim] =
          mlir::getAffineConstantExpr(step.index_step, context) *
          mlir::getAffineDimExpr(time_dim, context);
    }
    return result;
  }

  /// Drop time dimension `time_dim`. The remaining dimensions keep their
  /// relative order and are renumbered by `offsets()`.
  void eraseDim(unsigned time_dim) {
    extents.erase(extents.begin() + time_dim);
    steps.erase(steps.begin() + time_dim);
  }
};

/// Describe how `sizes` is traversed over time. Every non-unit dimension
/// except the innermost contributes a time dimension stepping by one; the
/// innermost contributes one stepping by a whole vector, and only when it
/// holds more than one. With a 64-lane vector:
///
///   sizes           extents      offsets           time_set
///   [1, 256, 64]    [256]        (0, d0, 0)        (d0) : 0 <= d0 <= 255
///   [1, 1, 128]     [2]          (0, 0, 64 * d0)   (d0) : 0 <= d0 <= 1
///   [2, 4, 8, 64]   [2, 4, 8]    (d0, d1, d2, 0)   3 dims of those extents
///   [1, 64]         []           (0, 0)            nothing walked
///
/// The last row is a transfer that fits in one vector: the offsets are already
/// the all-zero map, and the caller supplies the single pinned time step.
TransferTimeDims describeTransferTimeDims(llvm::ArrayRef<int64_t> sizes,
                                          int64_t lanes) {
  TransferTimeDims dims(sizes.size());
  auto addDim = [&](unsigned pos, int64_t extent, int64_t index_step) {
    dims.steps.push_back({pos, index_step});
    dims.extents.push_back(extent);
  };
  for (unsigned i = 0; i + 1 < sizes.size(); ++i) {
    if (sizes[i] != 1) addDim(i, sizes[i], /*index_step=*/1);
  }
  if (const int64_t vectors = sizes.back() / lanes; vectors > 1) {
    addDim(sizes.size() - 1, vectors, /*index_step=*/lanes);
  }
  return dims;
}

/// The coefficient of each dimension of `map`, which must have one result, or
/// failure if that result is not a linear function of those dimensions and so
/// has no one coefficient per dimension. A symbol, `floordiv`, `ceildiv` or
/// `mod` in the result is what makes it fail.
///
/// Flattening the map into a Presburger relation is what makes those cases
/// detectable. Evaluating the map at sample points cannot tell them apart from
/// a genuine coefficient.
mlir::FailureOr<llvm::SmallVector<int64_t>> coefficientsIfLinearInDims(
    mlir::AffineMap map) {
  assert(map.getNumResults() == 1 && "expected a one-result map");
  // Not known until runtime, so never a static distance.
  if (map.getNumSymbols() != 0) {
    return mlir::failure();
  }

  mlir::presburger::IntegerRelation relation(
      mlir::presburger::PresburgerSpace::getRelationSpace());
  // Fails for a semi-affine map, which cannot be flattened at all.
  if (mlir::failed(mlir::affine::getRelationFromMap(map, relation))) {
    return mlir::failure();
  }

  // Flattening emits inequalities only to bound the local variables it
  // introduces for `floordiv`, `ceildiv` and `mod`, so any inequality means
  // the result is not linear in the dimensions.
  if (relation.getNumInequalities() != 0) {
    return mlir::failure();
  }
  assert(relation.getNumLocalVars() == 0 &&
         "a local variable is always bounded by inequalities");

  // What is left is the one equality getRelationFromMap states per result,
  //
  //   coefficient_0 * d_0 + ... + coefficient_n * d_n - r + constant == 0
  //
  // whose dimension columns are the answer. Its constant column is the map's
  // base offset, which a distance does not depend on.
  if (relation.getNumEqualities() != 1) {
    return mlir::failure();
  }
  const unsigned num_dims = map.getNumDims();
  // The -1 that equates the flattened expression to `r`. Any other value would
  // scale or flip every coefficient taken below.
  if (relation.atEq64(0, /*result column=*/num_dims) != -1) {
    return mlir::failure();
  }
  llvm::SmallVector<int64_t> coefficients;
  for (unsigned dim = 0; dim < num_dims; ++dim) {
    coefficients.push_back(relation.atEq64(0, dim));
  }
  return coefficients;
}

/// The distance, in elements of the underlying linear memory, between
/// consecutive indices of each dimension of `memref`. A memory view states its
/// own linearization, which need not be the row-major layout implied by the
/// memref shape, so prefer it; the type's own layout is only the fallback.
/// Fails when the layout is not a static linear function of the indices.
mlir::FailureOr<llvm::SmallVector<int64_t>> getElementStrides(
    mlir::Value memref) {
  auto memref_type = llvm::cast<mlir::MemRefType>(memref.getType());
  const unsigned rank = memref_type.getRank();
  if (auto view =
          memref.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>()) {
    auto layout = view.getLayoutMap();
    if (layout.getNumDims() != rank || layout.getNumResults() != 1) {
      return mlir::failure();
    }
    return coefficientsIfLinearInDims(layout);
  }
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(memref_type.getStridesAndOffset(strides, offset))) {
    return mlir::failure();
  }
  if (llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
    return mlir::failure();
  }
  return strides;
}

/// Extend `map` with one trailing dimension that advances `step.memref_dim` by
/// `step.index_step` indices per unit. Appending a loop induction variable to
/// the subscript operands then drives that dimension from the loop instead of
/// from the time axis.
mlir::AffineMap foldStepIntoSubscripts(mlir::MLIRContext* context,
                                       mlir::AffineMap map,
                                       TransferTimeStep step) {
  llvm::SmallVector<mlir::AffineExpr> results(map.getResults());
  results[step.memref_dim] =
      results[step.memref_dim] +
      mlir::getAffineConstantExpr(step.index_step, context) *
          mlir::getAffineDimExpr(map.getNumDims(), context);
  return mlir::AffineMap::get(map.getNumDims() + 1, map.getNumSymbols(),
                              results, context);
}

/// Emit a self-sync (dataflow.sync_send) before `indirect_transfer`, hoisted as
/// far out of enclosing loops as possible without crossing a loop block that
/// also encloses the IAB fill (`fill_op`). E.g.
/// scf.for {
///   agen.composite_load_and_store ... <IAB fill>
/// }
/// <self-sync here>
/// scf.for {
///   agen.composite_load_and_store ... <Indirect load/store>
/// }
///
/// `fill_op` must be non-null; callers must only call this when a fill exists.
/// The fill's ancestor blocks are collected once (O(depth)) and used as an
/// O(1) membership test while walking up from `indirect_transfer` to find the
/// hoist boundary.
static void emitSelfSyncIndirect(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::ktdf::IndDataTransferOp indirect_transfer, mlir::Operation* fill_op,
    mlir::dataflow::ProgramUnitOp program_unit,
    const ResourceToUnits& components) {
  assert(fill_op && "emitSelfSyncIndirect requires a non-null fill_op");
  mlir::Operation* insertion_op = indirect_transfer.getOperation();
  // Collect the set of blocks enclosing fill_op to find the hoist boundary.
  llvm::DenseSet<mlir::Block*> fill_ancestor_blocks;
  for (mlir::Operation* p = fill_op->getParentOp(); p; p = p->getParentOp())
    fill_ancestor_blocks.insert(p->getBlock());

  mlir::Operation* cursor = indirect_transfer->getParentOp();
  while (cursor && !mlir::isa<mlir::dataflow::ProgramUnitOp>(cursor)) {
    if (mlir::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(cursor)) {
      // cursor is a common ancestor of op and fill_op so should not hoist
      // the self-sync outside of cursor.
      if (fill_ancestor_blocks.count(cursor->getBlock())) break;
      insertion_op = cursor;
    }
    cursor = cursor->getParentOp();
  }

  llvm::SmallVector<mlir::Value, 4> self_units(program_unit.getUnits().begin(),
                                               program_unit.getUnits().end());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertion_op);
  mlir::Value self_unit =
      createQueryMapForComponent(rewriter, program_unit, self_units, loc);
  mlir::dataflow::SyncSendOp::create(
      rewriter, loc, self_unit,
      /*dbgName=*/nullptr,
      /*wait_immediately_for_async_transfers=*/rewriter.getBoolAttr(true));
}

/// Pattern to lower ktdf.ind_data_transfer to
/// agen.composite_indirect_load_and_store.
///
/// Scatter mode (ind_dst present, ind_src absent):
///   dir_src is a local memref; dir_dst is a memref in global memory.
///   The IAB entry at ind_dst_index drives the destination base address.
///   The body is empty (just agen.yield).
///
/// Gather mode (ind_src present, ind_dst absent):
///   dir_src is a memref in global memory; dir_dst is either a memref or a
///   !ktdf.fifo.slot. The IAB entry at ind_src_index drives the source base
///   address. When dir_dst is a FIFO slot, the body emits dataflow.send.
struct LowerIndDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::IndDataTransferOp> {
  LowerIndDataTransferPattern(mlir::MLIRContext* context,
                              const ResourceToUnits& components)
      : OpRewritePattern(context), components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::IndDataTransferOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();
    const auto loc = op.getLoc();

    const bool is_gather = op.isGather();
    const bool is_scatter = op.isScatter();
    assert((is_gather ^ is_scatter) && "exactly one of gather/scatter");

    auto dir_src = op.getDirSrc();
    auto dir_dst = op.getDirDst();

    // dir_src must be a memref in both modes (op verifier guarantees this for
    // gather; scatter also requires a concrete source).
    if (!mlir::isa<mlir::MemRefType>(dir_src.getType())) {
      op.emitError(
          "ind_data_transfer lowering: dir_src must be a memref; FIFO "
          "dir_src is not yet supported");
      return mlir::failure();
    }
    auto dir_src_memref_type = mlir::cast<mlir::MemRefType>(dir_src.getType());

    const bool dst_is_fifo =
        mlir::isa<mlir::ktdf::FifoSlotType>(dir_dst.getType());
    if (is_scatter && dst_is_fifo) {
      op.emitError(
          "ind_data_transfer lowering: scatter mode requires dir_dst to be "
          "a memref");
      return mlir::failure();
    }

    auto static_src_sizes_attr = op.getStaticDirSrcSizes();
    auto static_dst_sizes_attr = op.getStaticDirDstSizes();
    if (!static_src_sizes_attr || !static_dst_sizes_attr) {
      op.emitError(
          "ind_data_transfer lowering: dynamic sizes are not yet supported");
      return mlir::failure();
    }
    llvm::SmallVector<int64_t> src_sizes(*static_src_sizes_attr);
    llvm::SmallVector<int64_t> dst_sizes(*static_dst_sizes_attr);

    auto elem_type = dir_src_memref_type.getElementType();

    const auto throttle = getThrottle(op);
    if (!throttle) {
      return rewriter.notifyMatchFailure(op, "unable to determine throttle");
    }

    const int64_t total_src = [&] {
      int64_t t = 1;
      for (int64_t s : src_sizes) t *= s;
      return t;
    }();
    const int64_t iv_lanes = std::min(total_src, *throttle);
    auto load_iv_type = mlir::VectorType::get({iv_lanes}, elem_type);

    // load_set / load_order: per-vector footprint on the dir_src side.
    llvm::SmallVector<int64_t> load_sizes(src_sizes.size(), 1);
    load_sizes.back() = iv_lanes;
    auto load_set = scheduler::buildIntegerSetFromSizes(ctx, load_sizes);
    auto load_order =
        mlir::AffineMap::getMultiDimIdentityMap(load_sizes.size(), ctx);

    // store_set / store_order: per-vector footprint on the dir_dst side.
    llvm::SmallVector<int64_t> store_sizes;
    if (!dst_is_fifo) {
      store_sizes.assign(dst_sizes.size(), 1);
      store_sizes.back() = iv_lanes;
    } else {
      store_sizes = load_sizes;
    }
    auto store_set = scheduler::buildIntegerSetFromSizes(ctx, store_sizes);
    auto store_order =
        mlir::AffineMap::getMultiDimIdentityMap(store_sizes.size(), ctx);

    // Time set / order / addr maps.
    TransferTimeDims src_time_dims =
        describeTransferTimeDims(src_sizes, *throttle);
    llvm::SmallVector<int64_t> time_extents = src_time_dims.extents;
    if (time_extents.empty()) time_extents.push_back(1);
    const unsigned num_time_dims = time_extents.size();

    auto time_set = scheduler::buildIntegerSetFromSizes(ctx, time_extents);
    auto time_order =
        mlir::AffineMap::getMultiDimIdentityMap(num_time_dims, ctx);

    auto load_direct_time_addr_map =
        mlir::AffineMap::get(num_time_dims, 0, src_time_dims.offsets(ctx), ctx);

    // Indirect time addr map: the IAB is indexed one entry per time step
    // (each entry is one address, so the step size is 1). Computed the same
    // way as the direct maps: describeTransferTimeDims on the IAB shape with
    // lanes=1.
    auto empty_map = mlir::AffineMap::get(0, 0, {}, ctx);
    mlir::AffineMap load_indirect_time_addr_map = empty_map;
    mlir::AffineMap store_indirect_time_addr_map = empty_map;
    if (is_gather) {
      auto iab_memref_type =
          mlir::cast<mlir::MemRefType>(op.getIndSrcMemref().getType());
      llvm::SmallVector<int64_t> iab_sizes(iab_memref_type.getShape());
      TransferTimeDims iab_time_dims =
          describeTransferTimeDims(iab_sizes, /*lanes=*/1);
      load_indirect_time_addr_map = mlir::AffineMap::get(
          num_time_dims, 0, iab_time_dims.offsets(ctx), ctx);
    } else {
      auto iab_memref_type =
          mlir::cast<mlir::MemRefType>(op.getIndDstMemref().getType());
      llvm::SmallVector<int64_t> iab_sizes(iab_memref_type.getShape());
      TransferTimeDims iab_time_dims =
          describeTransferTimeDims(iab_sizes, /*lanes=*/1);
      store_indirect_time_addr_map = mlir::AffineMap::get(
          num_time_dims, 0, iab_time_dims.offsets(ctx), ctx);
    }

    mlir::AffineMap store_direct_time_addr_map;
    if (!dst_is_fifo) {
      TransferTimeDims dst_time_dims =
          describeTransferTimeDims(dst_sizes, *throttle);
      store_direct_time_addr_map = mlir::AffineMap::get(
          num_time_dims, 0, dst_time_dims.offsets(ctx), ctx);
    } else {
      store_direct_time_addr_map =
          mlir::AffineMap::get(num_time_dims, 0,
                               llvm::SmallVector<mlir::AffineExpr>{
                                   mlir::getAffineConstantExpr(0, ctx)},
                               ctx);
    }

    // dir_src access map.
    auto dir_src_map =
        op.getDirSrcMap().value_or(mlir::AffineMap::getMultiDimIdentityMap(
            dir_src_memref_type.getRank(), ctx));

    // dir_dst access map and memref value.
    mlir::Value dir_dst_memref;
    mlir::AffineMap dir_dst_map;
    if (!dst_is_fifo) {
      dir_dst_memref = dir_dst;
      auto dir_dst_memref_type =
          mlir::cast<mlir::MemRefType>(dir_dst.getType());
      dir_dst_map =
          op.getDirDstMap().value_or(mlir::AffineMap::getMultiDimIdentityMap(
              dir_dst_memref_type.getRank(), ctx));
    } else {
      // FIFO dst: pass dir_src as the placeholder direct_dst_memref so the
      // builder's mandatory-memref assertion is satisfied.  The actual data
      // path is expressed inside the body via dataflow.send.
      dir_dst_memref = dir_src;
      dir_dst_map = dir_src_map;
    }

    // Indirect memref values, their index operands, and access maps.
    // Use the map stored on the op if present; fall back to an identity map
    // whose rank matches the IAB memref.
    mlir::Value ind_src_memref;
    mlir::Value ind_dst_memref;
    mlir::Value ind_src_index;
    mlir::Value ind_dst_index;
    mlir::AffineMap ind_src_map;
    mlir::AffineMap ind_dst_map;
    if (is_gather) {
      ind_src_memref = op.getIndSrcMemref();
      ind_src_index = op.getIndSrcIndex();
      const auto ind_src_rank =
          mlir::cast<mlir::MemRefType>(ind_src_memref.getType()).getRank();
      ind_src_map = op.getIndSrcMap().value_or(
          mlir::AffineMap::getMultiDimIdentityMap(ind_src_rank, ctx));
    } else {
      ind_dst_memref = op.getIndDstMemref();
      ind_dst_index = op.getIndDstIndex();
      const auto ind_dst_rank =
          mlir::cast<mlir::MemRefType>(ind_dst_memref.getType()).getRank();
      ind_dst_map = op.getIndDstMap().value_or(
          mlir::AffineMap::getMultiDimIdentityMap(ind_dst_rank, ctx));
    }

    // Operands list (per builder contract):
    //   indirect_src_indices, direct_src_indices,
    //   indirect_dst_indices, direct_dst_indices,
    //   multicast_info (none), time_symbols (none).
    llvm::SmallVector<mlir::Value> operands;
    uint32_t num_ind_src_indices = 0;
    uint32_t num_ind_dst_indices = 0;

    if (ind_src_index) {
      operands.push_back(ind_src_index);
      num_ind_src_indices = 1;
    }
    llvm::SmallVector<mlir::Value> dir_src_indices(op.getDirSrcIndices());
    operands.append(dir_src_indices.begin(), dir_src_indices.end());
    if (ind_dst_index) {
      operands.push_back(ind_dst_index);
      num_ind_dst_indices = 1;
    }
    llvm::SmallVector<mlir::Value> dir_dst_indices(op.getDirDstIndices());
    if (!dst_is_fifo) {
      operands.append(dir_dst_indices.begin(), dir_dst_indices.end());
    } else {
      // FIFO dst: use dir_src indices as placeholder for direct_dst.
      operands.append(dir_src_indices.begin(), dir_src_indices.end());
    }
    const uint32_t num_dir_src_indices =
        static_cast<uint32_t>(dir_src_indices.size());
    const uint32_t num_dir_dst_indices =
        dst_is_fifo ? num_dir_src_indices
                    : static_cast<uint32_t>(dir_dst_indices.size());

    // Resolve the enclosing program_unit (needed for self-sync and FIFO dest).
    auto program_unit = op->getParentOfType<mlir::dataflow::ProgramUnitOp>();

    // Resolve the FIFO destination unit for gather-to-FIFO before creating
    // the composite op; the send is inserted into the body afterwards because
    // CompositeIndirectLoadAndStoreOp::build does not invoke the bodyBuilder
    // callback.
    mlir::Value fifo_dest_unit;
    if (dst_is_fifo) {
      auto dst_fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(dir_dst.getType());
      auto dest_unit_result = resolveUnitFromFifoAttr(
          dst_fifo_slot_type.getDest(), components_, rewriter, program_unit,
          loc, op.getOperation());
      if (mlir::failed(dest_unit_result)) return mlir::failure();
      fifo_dest_unit = *dest_unit_result;
    }

    // Emit a self-sync before the indirect transfer only when there is a
    // DataTransferOp whose destination is the IAB.
    // The sync is hoisted as far out of enclosing loops as possible without
    // crossing the fill. If the IAB has no fill, no sync is needed.
    {
      mlir::Value iab_memref =
          is_gather ? op.getIndSrcMemref() : op.getIndDstMemref();
      mlir::Operation* fill_op = nullptr;
      for (mlir::Operation* user : iab_memref.getUsers()) {
        if (auto dt = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(user)) {
          if (dt.getDestination() == iab_memref) {
            fill_op = user;
            break;
          }
        }
      }
      if (fill_op)
        emitSelfSyncIndirect(rewriter, loc, op, fill_op, program_unit,
                             components_);
    }

    auto composite_op = mlir::agen::CompositeIndirectLoadAndStoreOp::create(
        rewriter, loc,
        /*indirect_src_memref=*/ind_src_memref,
        /*direct_src_memref=*/dir_src,
        /*indirect_dst_memref=*/ind_dst_memref,
        /*direct_dst_memref=*/dir_dst_memref,
        /*dbgName=*/nullptr,
        /*indirect_src_map=*/
        is_gather ? ind_src_map : mlir::AffineMap::get(0, 0, {}, ctx),
        /*direct_src_map=*/dir_src_map,
        /*indirect_dst_map=*/
        is_scatter ? ind_dst_map : mlir::AffineMap::get(0, 0, {}, ctx),
        /*direct_dst_map=*/dir_dst_map,
        /*operands=*/operands,
        /*type=*/load_iv_type,
        /*load_set=*/load_set,
        /*load_order=*/load_order,
        /*store_set=*/store_set,
        /*store_order=*/store_order,
        /*time_set=*/time_set,
        /*time_order=*/time_order,
        /*load_indirect_time_addr_map=*/load_indirect_time_addr_map,
        /*load_direct_time_addr_map=*/load_direct_time_addr_map,
        /*store_indirect_time_addr_map=*/store_indirect_time_addr_map,
        /*store_direct_time_addr_map=*/store_direct_time_addr_map,
        /*num_ind_src_memref_indices=*/num_ind_src_indices,
        /*num_dir_src_memref_indices=*/num_dir_src_indices,
        /*num_ind_dst_memref_indices=*/num_ind_dst_indices,
        /*num_dir_dst_memref_indices=*/num_dir_dst_indices,
        /*num_multicast_info=*/0,
        /*num_time_symbols=*/0,
        /*bodyBuilder=*/nullptr);

    // Insert dataflow.send into the body before the terminator for
    // gather-to-FIFO transfers (the builder callback is not invoked by
    // CompositeIndirectLoadAndStoreOp::build).
    if (dst_is_fifo) {
      mlir::Block& body = composite_op.getRegion().front();
      mlir::Value load_iv = composite_op.getLoadInductionVar();
      mlir::OpBuilder body_builder(body.getTerminator());
      mlir::dataflow::SendOp::create(body_builder, loc, fifo_dest_unit, load_iv,
                                     /*dir=*/nullptr,
                                     /*dbgName=*/nullptr);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

 private:
  const ResourceToUnits& components_;
};

/// Pattern to lower ktdf.data_transfer operations
struct LowerDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  LowerDataTransferPattern(mlir::MLIRContext* context,
                           const ResourceToUnits& components)
      : OpRewritePattern(context), components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::DataTransferOp data_transfer_op,
      mlir::PatternRewriter& rewriter) const override {
    auto src = data_transfer_op.getSource();
    auto dst = data_transfer_op.getDestination();

    bool src_is_fifo = data_transfer_op.isSourceFifo();
    bool dst_is_fifo = data_transfer_op.isDestFifo();

    // Extract FIFO types if applicable
    // Determine transfer type based on source and destination
    auto transfer_type_or = getDataTransferType(src_is_fifo, dst_is_fifo);
    if (mlir::failed(transfer_type_or)) {
      data_transfer_op.emitError(
          "Unsupported data transfer: FIFO to FIFO transfers are not allowed");
      return mlir::failure();
    }
    auto transfer_type = *transfer_type_or;

    // Get source and destination indices
    auto src_indices = data_transfer_op.getSourceIndices();
    auto dst_indices = data_transfer_op.getDestIndices();

    // Get static sizes
    assert(data_transfer_op.hasAllStaticSourceSizes() &&
           "Expected static source sizes");
    assert(data_transfer_op.hasAllStaticDestSizes() &&
           "Expected static dest sizes");

    auto src_static_sizes = *data_transfer_op.getStaticSourceSizes();
    auto dst_static_sizes = *data_transfer_op.getStaticDestSizes();

    // Calculate total elements from static sizes
    int64_t src_total_elements = 1;
    for (int64_t size : src_static_sizes) {
      src_total_elements *= size;
    }

    int64_t dst_total_elements = 1;
    for (int64_t size : dst_static_sizes) {
      dst_total_elements *= size;
    }

    // For splat/pad transfers the source is smaller than the destination —
    // the hardware replicates or zero-pads to fill the vector. Skip the
    // equality check and use the destination size as the transfer width.
    // For lane0 transfers the destination is one element: the full FIFO slot
    // is received and only its lane 0 is stored, so the source size is the
    // transfer width.
    auto transfer_mode_attr = llvm::dyn_cast_if_present<mlir::StringAttr>(
        data_transfer_op->getDiscardableAttr("transfer_mode"));
    llvm::StringRef transfer_mode =
        transfer_mode_attr ? transfer_mode_attr.getValue() : "";
    bool is_broadcast_transfer =
        transfer_mode == "splat" || transfer_mode == "pad";
    bool is_lane0_transfer = transfer_mode == "lane0";

    if (is_broadcast_transfer) {
      if (src_total_elements > dst_total_elements) {
        data_transfer_op.emitError(
            "source total elements must not exceed destination for "
            "splat/pad transfer");
        return mlir::failure();
      }
    }

    if (is_lane0_transfer) {
      if (!src_is_fifo || dst_is_fifo) {
        data_transfer_op.emitError(
            "lane0 transfer must be from a FIFO slot to a memref");
        return mlir::failure();
      }
      if (dst_total_elements != 1) {
        data_transfer_op.emitError(
            "lane0 transfer must have a one-element destination");
        return mlir::failure();
      }
    }

    int64_t total_elements =
        is_lane0_transfer ? src_total_elements : dst_total_elements;

    // Get element type (from memref or FIFO slot)
    mlir::Type elem_type;
    if (src_is_fifo) {
      elem_type =
          llvm::cast<mlir::ktdf::FifoSlotType>(src.getType()).getElementType();
    } else {
      elem_type = llvm::cast<mlir::MemRefType>(src.getType()).getElementType();
    }

    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    // Handle different transfer types
    switch (transfer_type) {
      case DataTransferType::kLoadAndStore: {
        // Both are memrefs
        auto src_memref = src;
        auto dst_memref = dst;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned src_num_dims = src_memref_type.getRank();
        unsigned dst_num_dims = dst_memref_type.getRank();

        auto src_map = data_transfer_op.getSourceMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(src_num_dims,
                                                    rewriter.getContext()));
        auto dst_map = data_transfer_op.getDestMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims,
                                                    rewriter.getContext()));

        return lowerAsLoadAndStore(
            rewriter, data_transfer_op, src_memref, dst_memref, src_indices,
            dst_indices, src_static_sizes, dst_static_sizes, src_num_dims,
            dst_num_dims, vector_type, src_map, dst_map);
      }

      case DataTransferType::kLoadAndSend: {
        // Source is memref, destination is FIFO
        auto src_memref = src;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        unsigned num_dims = src_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto src_map = data_transfer_op.getSourceMap().value_or(identity_map);

        auto dst_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(dst.getType());
        return lowerAsLoadAndSend(rewriter, data_transfer_op, src_memref,
                                  src_indices, src_static_sizes, num_dims,
                                  vector_type, src_map, dst_fifo_slot_type,
                                  is_broadcast_transfer, src_total_elements);
      }

      case DataTransferType::kReceiveAndStore: {
        // Source is FIFO, destination is memref
        auto dst_memref = dst;
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned num_dims = dst_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto dst_map = data_transfer_op.getDestMap().value_or(identity_map);

        auto src_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(src.getType());
        return lowerAsReceiveAndStore(rewriter, data_transfer_op, dst_memref,
                                      dst_indices, dst_static_sizes, num_dims,
                                      vector_type, dst_map, src_fifo_slot_type,
                                      is_lane0_transfer, dst_total_elements);
      }
    }

    return mlir::failure();
  }

 private:
  const ResourceToUnits& components_;

  /// Lower as CompositeLoadAndStore.
  ///
  /// An AGEN composite transfer moves at most one hardware vector per time
  /// step, so a transfer wider than that has to walk the remaining elements
  /// over AGEN time dimensions instead of widening `load_iv`.
  mlir::LogicalResult lowerAsLoadAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::Value dst_memref, mlir::ValueRange src_indices,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, unsigned src_num_dims,
      unsigned dst_num_dims, mlir::VectorType vector_type,
      mlir::AffineMap src_map, mlir::AffineMap dst_map) const {
    auto* context = rewriter.getContext();
    const int64_t total = vector_type.getNumElements();

    const auto throttle = getThrottle(data_transfer_op);
    if (!throttle) {
      return rewriter.notifyMatchFailure(data_transfer_op,
                                         "unable to determine throttle");
    }

    // Sizes describing the elements covered by one AGEN vector transfer, and
    // the dimensions (if any) walked over time to cover the rest. Narrowed
    // below when the request is wider than one hardware vector.
    llvm::SmallVector<int64_t> load_sizes(src_static_sizes);
    llvm::SmallVector<int64_t> store_sizes(dst_static_sizes);
    mlir::VectorType load_iv_type = vector_type;
    TransferTimeDims src_time_dims(src_static_sizes.size());
    TransferTimeDims dst_time_dims(dst_static_sizes.size());
    // Set when one time dimension has to leave the time axis for an enclosing
    // loop; see below.
    std::optional<unsigned> loop_time_dim;

    if (total > *throttle) {
      if (src_static_sizes.empty() || dst_static_sizes.empty()) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the throttle of " << *throttle
            << " but has no dimensions to split";
        return mlir::failure();
      }

      if (src_static_sizes.back() % *throttle != 0 ||
          dst_static_sizes.back() % *throttle != 0) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the throttle of " << *throttle
            << "; splitting requires the innermost source and destination "
               "sizes to be a multiple of the throttle, but they are "
            << src_static_sizes.back() << " and " << dst_static_sizes.back();
        return mlir::failure();
      }

      src_time_dims = describeTransferTimeDims(src_static_sizes, *throttle);
      dst_time_dims = describeTransferTimeDims(dst_static_sizes, *throttle);

      // Only the extents are compared, not positions or coefficients: the
      // two sides may reach the same walk through different shapes, e.g.
      // src [2, 64] and dst [1, 128] with 64 lanes both walk 2 steps.
      if (src_time_dims.extents != dst_time_dims.extents) {
        data_transfer_op.emitError()
            << "source and destination walked dimensions must match to "
               "split a transfer of "
            << total << " elements across multiple vectors";
        return mlir::failure();
      }

      load_iv_type =
          mlir::VectorType::get({*throttle}, vector_type.getElementType());
      load_sizes.assign(src_static_sizes.size(), 1);
      load_sizes.back() = *throttle;
      store_sizes.assign(dst_static_sizes.size(), 1);
      store_sizes.back() = *throttle;

      // A time dimension is one step count shared by both sides, so it is only
      // realizable when both sides move the same distance per step: one count
      // cannot stand for two different strides. Matching extents do not imply
      // matching distances — the sides may reach the same walk through
      // different shapes and layouts — so compare the distances.
      auto src_strides = getElementStrides(src_memref);
      auto dst_strides = getElementStrides(dst_memref);
      if (mlir::succeeded(src_strides) && mlir::succeeded(dst_strides)) {
        auto distance = [](TransferTimeStep step,
                           llvm::ArrayRef<int64_t> strides) {
          return step.index_step * strides[step.memref_dim];
        };
        llvm::SmallVector<unsigned> divergent;
        for (unsigned i = 0, e = src_time_dims.extents.size(); i < e; ++i) {
          if (src_time_dims.extents[i] == 1) continue;
          if (distance(src_time_dims.steps[i], *src_strides) !=
              distance(dst_time_dims.steps[i], *dst_strides)) {
            divergent.push_back(i);
          }
        }
        // A dimension whose distances disagree can still be walked by an
        // enclosing loop: its step then lands in the memref subscripts, where
        // each side applies its own layout. Several such dimensions would need
        // a nesting order among them, which nothing here determines.
        if (divergent.size() > 1) {
          data_transfer_op.emitError()
              << "source and destination advance by different distances in "
              << divergent.size() << " walked dimensions of a transfer of "
              << total << " elements; only one such dimension can be resolved";
          return mlir::failure();
        }
        if (divergent.size() == 1) loop_time_dim = divergent.front();
      }
    }

    // Subscript operands, gaining the loop induction variable when a time
    // dimension moves to an enclosing loop.
    llvm::SmallVector<mlir::Value> load_indices(src_indices);
    llvm::SmallVector<mlir::Value> store_indices(dst_indices);

    if (loop_time_dim) {
      const auto loc = data_transfer_op.getLoc();
      auto lower = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto upper = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, src_time_dims.extents[*loop_time_dim]);
      auto step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop = mlir::scf::ForOp::create(rewriter, loc, lower, upper, step);

      src_map = foldStepIntoSubscripts(context, src_map,
                                       src_time_dims.steps[*loop_time_dim]);
      load_indices.push_back(loop.getInductionVar());
      dst_map = foldStepIntoSubscripts(context, dst_map,
                                       dst_time_dims.steps[*loop_time_dim]);
      store_indices.push_back(loop.getInductionVar());

      src_time_dims.eraseDim(*loop_time_dim);
      dst_time_dims.eraseDim(*loop_time_dim);

      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Build src_load_set / dst_store_set from the per-vector sizes.
    auto src_load_set =
        scheduler::buildIntegerSetFromSizes(context, load_sizes);
    auto dst_store_set =
        scheduler::buildIntegerSetFromSizes(context, store_sizes);

    // load_order and store_order must match their respective set
    // dimensionality, which is the number of per-vector sizes (load_sizes /
    // store_sizes), not the rank of the original source/destination memref.
    auto load_order =
        mlir::AffineMap::getMultiDimIdentityMap(load_sizes.size(), context);
    auto store_order =
        mlir::AffineMap::getMultiDimIdentityMap(store_sizes.size(), context);

    // Time dimensions: a single pinned step for a transfer of at most one
    // vector, one dimension per walked dimension otherwise.
    llvm::SmallVector<int64_t> time_extents = src_time_dims.extents;
    if (time_extents.empty()) time_extents.push_back(1);
    const unsigned num_time_dims = time_extents.size();
    auto time_set = scheduler::buildIntegerSetFromSizes(context, time_extents);
    // time_order: identity. Correct because the walked dimensions above are
    // listed slowest-varying first, so time dimension 0 is already the
    // outermost and the last time dimension the innermost/fastest-varying.
    auto time_order =
        mlir::AffineMap::getMultiDimIdentityMap(num_time_dims, context);
    auto load_addr_map = mlir::AffineMap::get(
        num_time_dims, 0, src_time_dims.offsets(context), context);
    auto store_addr_map = mlir::AffineMap::get(
        num_time_dims, 0, dst_time_dims.offsets(context), context);

    // Create the CompositeLoadAndStoreOp
    mlir::agen::CompositeLoadAndStoreOp::create(
        rewriter, data_transfer_op.getLoc(), src_memref, dst_memref,
        /*dbgName=*/nullptr, src_map, load_indices, dst_map, store_indices,
        src_load_set, load_order, dst_store_set, store_order, {}, time_set,
        time_order, load_addr_map, store_addr_map, load_iv_type);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as vector_load and send (L1 to FIFO)
  mlir::LogicalResult lowerAsLoadAndSend(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::ValueRange src_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap src_map,
      mlir::ktdf::FifoSlotType dst_fifo_slot_type, bool is_splat,
      int64_t src_total_elements) const {
    // Build load_set from source sizes
    auto load_set =
        buildIntegerSetFromSizes(rewriter.getContext(), src_static_sizes);

    // Build load_order
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // When splat: load only src_total_elements, then shuffle to full width.
    if (is_splat) {
      if (vector_type.getNumElements() % src_total_elements != 0) {
        data_transfer_op.emitError(
            "dst_total_elements must be divisible by src_total_elements "
            "for splat transfer");
        return mlir::failure();
      }
    }
    auto load_type = is_splat
                         ? mlir::VectorType::get({src_total_elements},
                                                 vector_type.getElementType())
                         : vector_type;

    // Create vector_load operation
    auto vector_load_op = mlir::agen::VectorLoadOp::create(
        rewriter, data_transfer_op.getLoc(), load_type, src_memref,
        /*dbgName=*/nullptr, src_map, src_indices, load_set, load_order);

    // For splat: broadcast loaded vector to full destination width.
    mlir::Value send_value = vector_load_op.getResult();
    if (is_splat) {
      send_value =
          insertSplatShuffle(rewriter, data_transfer_op.getLoc(), send_value,
                             src_total_elements, vector_type.getNumElements());
    }

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto dest_unit_result = resolveUnitFromFifoAttr(
        dst_fifo_slot_type.getDest(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(dest_unit_result)) {
      return mlir::failure();
    }
    mlir::Value dest_unit = *dest_unit_result;

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, data_transfer_op.getLoc(),
                                   dest_unit, send_value,
                                   /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as receive and vector_store (FIFO to L1). When is_lane0, the
  /// received vector is narrowed to its lane 0, which is stored at the
  /// destination indices.
  mlir::LogicalResult lowerAsReceiveAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value dst_memref,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> dst_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap dst_map,
      mlir::ktdf::FifoSlotType src_fifo_slot_type, bool is_lane0,
      int64_t dst_total_elements) const {
    // Build store_set from destination sizes
    auto store_set =
        buildIntegerSetFromSizes(rewriter.getContext(), dst_static_sizes);

    // Build store_order
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto src_unit_result = resolveUnitFromFifoAttr(
        src_fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(src_unit_result)) {
      return mlir::failure();
    }
    mlir::Value src_unit = *src_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, data_transfer_op.getLoc(), vector_type, src_unit,
        /*dbgName=*/nullptr);

    // For lane0: keep only lane 0.
    mlir::Value store_value = receive_op.getData();
    if (is_lane0) {
      store_value = insertLeadingLanesShuffle(
          rewriter, data_transfer_op.getLoc(), store_value, dst_total_elements);
    }

    // Create vector_store operation
    mlir::agen::VectorStoreOp::create(
        rewriter, data_transfer_op.getLoc(), store_value, dst_memref,
        /*dbgName=*/nullptr, dst_map, dst_indices, store_set, store_order);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateDataTransferLoweringPatterns(
    mlir::RewritePatternSet& patterns, const ResourceToUnits& components) {
  patterns.add<LowerIndDataTransferPattern>(patterns.getContext(), components);
  patterns.add<LowerDataTransferPattern>(patterns.getContext(), components);
}
