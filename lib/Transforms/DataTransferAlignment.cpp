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
// DataTransferAlignment: make strided copies between global views legal.
//
// A transpose or permute of stick-tiled tensors reaches this pass as a copy
// from a global view x to a global view y through two per-core staging
// buffers, A and B:
//
//   stage L3LU:  x  -> A     one stick of x per transfer
//   stage ...:   A  -> B     a nested pipeline moving one stick at a time
//   stage L3SU:  B  -> y     one stick of y per transfer
//
// Every transfer carries one index per view dim, and x and y index the same
// loop dims in the same order. A stick is a run of `block` contiguous elements
// (128 bytes: 64 fp16). Let p be x's stride-1 dim and q y's. y is row-major,
// so q is its last dim.
//
//   - p == q: x's sticks are y's sticks. Each transfer already moves a whole
//     contiguous stick; nothing is widened.
//   - p != q: a stick of x runs along p, a stick of y along q. The pass widens
//     the copy to `block` x `block` tiles: the loop over dim p is pinned to one
//     stick, the load and store move a whole tile, and the nested pipeline
//     transposes each tile element by element (splat out of A, extract into
//     B). That transpose is the only one: the load fills row a of A with
//     stick a of x, so it reads from x_t, a second view of x's memory with
//     dims p and last swapped, whose last dim has stride 1. x itself is not
//     rewritten.
//
// Finally every global view is listed in decreasing-stride order, with the
// maps of the transfers on it permuted to match. That only relabels which
// index slot addresses which dim; every address is unchanged. The re-stick is
// correct without it; the device compiler requires views in that order.
//
// Ordering: after stage-coarsening + canonicalize, before double-buffering.
//
//===----------------------------------------------------------------------===//

#include <limits>
#include <numeric>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "data-transfer-alignment"
#define DEBUG_TYPE PASS_NAME

namespace scheduler {
#define GEN_PASS_DEF_DATATRANSFERALIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

//===----------------------------------------------------------------------===//
// Views
//===----------------------------------------------------------------------===//

/// Whether `value` is a memref in a core's own scratchpad (a staging buffer).
bool isScratchpad(mlir::Value value,
                  const scheduler::arch_view::MemoryTree& memory_tree) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type) return false;
  auto kind =
      mlir::dyn_cast_or_null<mlir::ktdf_arch::KindAttr>(type.getMemorySpace());
  return kind && memory_tree.isPerCoreScratchPadMemory(kind);
}

/// The static shape and strides of the memref `value`; failure if it is not
/// a memref, its layout has no strides, or any of them is dynamic.
struct Layout {
  llvm::SmallVector<int64_t> sizes, strides;
};
mlir::FailureOr<Layout> layoutOf(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape()) return mlir::failure();
  Layout layout{llvm::to_vector(type.getShape()), {}};
  int64_t offset;
  if (mlir::failed(type.getStridesAndOffset(layout.strides, offset)) ||
      llvm::any_of(layout.strides, mlir::ShapedType::isDynamic))
    return mlir::failure();
  return layout;
}

/// The dim with stride 1 and more than one element, if any. A size-1 dim is
/// skipped: its stride never multiplies a non-zero index.
std::optional<int64_t> unitStrideDim(const Layout& layout) {
  for (auto [dim, size_stride] :
       llvm::enumerate(llvm::zip(layout.sizes, layout.strides))) {
    auto [size, stride] = size_stride;
    if (stride == 1 && size != 1) return dim;
  }
  return std::nullopt;
}

/// The dims of `view` listed by decreasing stride. A size-1 dim sorts first:
/// its index is always 0, so its position does not matter, and putting it
/// first keeps the non-unit dims in order. Ties keep their order.
llvm::SmallVector<int64_t> strideOrder(mlir::ktdp::ConstructMemoryViewOp view) {
  llvm::ArrayRef<int64_t> sizes = view.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = view.getStaticStrides();
  llvm::SmallVector<int64_t> order(sizes.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](int64_t a, int64_t b) {
    bool a_unit = sizes[a] == 1, b_unit = sizes[b] == 1;
    if (a_unit != b_unit) return a_unit;
    return !a_unit && strides[a] > strides[b];
  });
  return order;
}

template <typename T>
llvm::SmallVector<T> permuted(llvm::ArrayRef<T> values,
                              llvm::ArrayRef<int64_t> order) {
  return llvm::map_to_vector(order, [&](int64_t i) { return values[i]; });
}

/// `type` with its dims listed in `order`: each dim keeps its size and stride,
/// so every element keeps its address. The offset and memory space stay.
mlir::FailureOr<mlir::MemRefType> permutedMemRefType(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> order) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return mlir::failure();
  auto layout = mlir::StridedLayoutAttr::get(
      type.getContext(), offset, permuted(llvm::ArrayRef(strides), order));
  return mlir::MemRefType::get(permuted(type.getShape(), order),
                               type.getElementType(), layout,
                               type.getMemorySpace());
}

/// `set` with its dims listed in `order`: new dim i is old dim order[i].
mlir::IntegerSet permutedCoordinateSet(mlir::IntegerSet set,
                                       llvm::ArrayRef<int64_t> order) {
  mlir::MLIRContext* ctx = set.getContext();
  llvm::SmallVector<mlir::AffineExpr> rename(order.size());
  for (auto [new_pos, old_pos] : llvm::enumerate(order))
    rename[old_pos] = mlir::getAffineDimExpr(new_pos, ctx);
  return set.replaceDimsAndSymbols(rename, {}, order.size(), 0);
}

/// Lists the dims of `view` in `order`: its sizes, strides, coordinate set and
/// result type. The ops using its result are not touched.
void permuteViewOp(mlir::ktdp::ConstructMemoryViewOp view,
                   llvm::ArrayRef<int64_t> order, mlir::MemRefType new_type) {
  mlir::MLIRContext* ctx = view.getContext();
  view.setStaticSizesAttr(mlir::DenseI64ArrayAttr::get(
      ctx, permuted(view.getStaticSizes(), order)));
  view.setStaticStridesAttr(mlir::DenseI64ArrayAttr::get(
      ctx, permuted(view.getStaticStrides(), order)));
  view.setCoordinateSetAttr(mlir::IntegerSetAttr::get(
      permutedCoordinateSet(view.getCoordinateSet().getValue(), order)));
  view.getResult().setType(new_type);
}

/// The index constant `value` at the start of `block`, reusing one already
/// there so each value is defined once.
mlir::Value indexConstant(mlir::Block* block, int64_t value,
                          mlir::OpBuilder& builder) {
  for (auto cst : block->getOps<mlir::arith::ConstantIndexOp>())
    if (cst.value() == value) return cst;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  return mlir::arith::ConstantIndexOp::create(
      builder, builder.getUnknownLoc(), value);
}

//===----------------------------------------------------------------------===//
// Transfers
//===----------------------------------------------------------------------===//

/// The editable operands of a ktdf.data_transfer. `rebuild` replaces the op
/// with one built from them, keeping its discardable attributes.
struct TransferParts {
  mlir::Value source, destination;
  mlir::AffineMap src_map, dst_map;
  llvm::SmallVector<mlir::Value> src_indices, dst_indices;
  llvm::SmallVector<mlir::OpFoldResult> src_sizes, dst_sizes;

  explicit TransferParts(mlir::ktdf::DataTransferOp op)
      : source(op.getSource()),
        destination(op.getDestination()),
        src_map(op.getSourceMap().value_or(mlir::AffineMap())),
        dst_map(op.getDestMap().value_or(mlir::AffineMap())),
        src_indices(op.getSourceIndices()),
        dst_indices(op.getDestIndices()),
        src_sizes(op.getMixedSourceSizes()),
        dst_sizes(op.getMixedDestSizes()) {}

  mlir::ktdf::DataTransferOp rebuild(mlir::ktdf::DataTransferOp op,
                                     mlir::OpBuilder& builder) const {
    builder.setInsertionPoint(op);
    auto replacement = mlir::ktdf::DataTransferOp::create(
        builder, op.getLoc(), source, src_map, src_indices, src_sizes,
        destination, dst_map, dst_indices, dst_sizes);
    for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
      replacement->setDiscardableAttr(attr.getName(), attr.getValue());
    op.erase();
    return replacement;
  }
};

mlir::OpFoldResult indexAttr(mlir::MLIRContext* ctx, int64_t value) {
  return mlir::IntegerAttr::get(mlir::IndexType::get(ctx), value);
}

/// The static value of transfer size `size`; nullopt if it is dynamic.
std::optional<int64_t> staticSize(mlir::OpFoldResult size) {
  if (auto attr = mlir::dyn_cast<mlir::Attribute>(size))
    return mlir::cast<mlir::IntegerAttr>(attr).getInt();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Finding a re-stick
//===----------------------------------------------------------------------===//

/// One strided copy x -> y found in a ktdf.pipeline, and what widening it
/// changes. Built by `findRestick`, consumed by `widenRestick`.
struct Restick {
  mlir::ktdf::DataTransferOp load;   // x -> A, in the first stage
  mlir::ktdf::DataTransferOp store;  // B -> y, in the last stage
  mlir::ktdf::PipelineOp nested;     // the A -> B per-stick pipeline
  int64_t p = 0;      // x's stride-1 dim; never the last dim
  int64_t rank = 0;   // rank of x's and y's views
  int64_t block = 0;  // elements in one stick: the load's last size
  /// The induction variable of the loop that walks dim p one element at a
  /// time (the point loop). It indexes slot p of the load either directly or
  /// through `linearize_index [tile_iv : tile], [point_iv : 1]`.
  mlir::Value point_iv;
  /// The tile size of dim p when its loop is tiled: a reserve_size
  /// placeholder. Null when the loop is not tiled.
  mlir::ktdf::TilingReserveSizeOp tile;
};

/// The ktdf.data_transfer ops directly inside `stage`, outside any nested
/// pipeline.
llvm::SmallVector<mlir::ktdf::DataTransferOp> stageTransfers(
    mlir::ktdf::StageOp stage) {
  llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
  stage->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(op)) return mlir::WalkResult::skip();
    if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op))
      transfers.push_back(transfer);
    return mlir::WalkResult::advance();
  });
  return transfers;
}

/// Whether the non-unit strides decrease from left to right. A size-1 dim is
/// skipped: its index is always 0.
bool stridesDecrease(llvm::ArrayRef<int64_t> sizes,
                     llvm::ArrayRef<int64_t> strides) {
  int64_t prev = std::numeric_limits<int64_t>::max();
  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    if (size == 1) continue;
    if (stride > prev) return false;
    prev = stride;
  }
  return true;
}

/// Whether `layout` is row-major: its non-unit strides decrease and its last
/// dim has stride 1.
bool isRowMajor(const Layout& layout) {
  return stridesDecrease(layout.sizes, layout.strides) &&
         !layout.strides.empty() && layout.strides.back() == 1;
}

/// The index value at memref dim `dim` of one transfer side, when the side's
/// map gives that dim a plain loop index; null otherwise.
mlir::Value indexAt(mlir::AffineMap map, mlir::ValueRange indices,
                    int64_t dim) {
  auto expr = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(dim));
  return expr ? indices[expr.getPosition()] : mlir::Value();
}

/// Whether index `a` of the load and index `b` of the store are the same
/// value. They are when they are one SSA value, equal constants, induction
/// variables paired in `ivs` (the loop each stage runs over one staging-buffer
/// dim), or results of the same op applied to such indices. Each stage
/// recomputes its own indices, so SSA identity alone is too strict.
bool sameIndex(mlir::Value a, mlir::Value b,
               const llvm::DenseMap<mlir::Value, mlir::Value>& ivs) {
  if (a == b) return true;
  if (auto it = ivs.find(a); it != ivs.end()) return it->second == b;
  auto ca = mlir::getConstantIntValue(a), cb = mlir::getConstantIntValue(b);
  if (ca || cb) return ca == cb;
  mlir::Operation *op_a = a.getDefiningOp(), *op_b = b.getDefiningOp();
  if (!op_a || !op_b || op_a->getName() != op_b->getName() ||
      op_a->getAttrDictionary() != op_b->getAttrDictionary() ||
      op_a->getNumOperands() != op_b->getNumOperands() ||
      op_a->getNumRegions() != 0 || op_b->getNumRegions() != 0 ||
      mlir::cast<mlir::OpResult>(a).getResultNumber() !=
          mlir::cast<mlir::OpResult>(b).getResultNumber())
    return false;
  return llvm::all_of(
      llvm::zip(op_a->getOperands(), op_b->getOperands()), [&](auto pair) {
        return sameIndex(std::get<0>(pair), std::get<1>(pair), ivs);
      });
}

/// Whether map result `a` over the load's indices and map result `b` over the
/// store's compute the same index: the same expression, with each pair of dims
/// at the same position naming the same index under `sameIndex`.
bool sameIndexExpr(mlir::AffineExpr a, mlir::ValueRange a_indices,
                   mlir::AffineExpr b, mlir::ValueRange b_indices,
                   const llvm::DenseMap<mlir::Value, mlir::Value>& ivs) {
  if (a.getKind() != b.getKind()) return false;
  if (auto dim_a = mlir::dyn_cast<mlir::AffineDimExpr>(a))
    return sameIndex(
        a_indices[dim_a.getPosition()],
        b_indices[mlir::cast<mlir::AffineDimExpr>(b).getPosition()], ivs);
  if (auto cst_a = mlir::dyn_cast<mlir::AffineConstantExpr>(a))
    return cst_a.getValue() ==
           mlir::cast<mlir::AffineConstantExpr>(b).getValue();
  auto bin_a = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(a);
  auto bin_b = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(b);
  return bin_a && bin_b &&
         sameIndexExpr(bin_a.getLHS(), a_indices, bin_b.getLHS(), b_indices,
                       ivs) &&
         sameIndexExpr(bin_a.getRHS(), a_indices, bin_b.getRHS(), b_indices,
                       ivs);
}

/// Looks in `pipeline` for a copy that needs re-sticking: a transfer from a
/// global view x whose last stride is not 1 into a scratchpad buffer.
/// Returns nullopt when there is none (nothing to do), failure after emitting
/// an error when there is one this pass cannot handle.
mlir::FailureOr<std::optional<Restick>> findRestick(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::arch_view::MemoryTree& memory_tree) {
  auto stages = llvm::to_vector(pipeline.getStages());
  if (stages.empty()) return std::optional<Restick>();

  // The load: x -> A in the first stage.
  mlir::ktdf::DataTransferOp load;
  for (auto transfer : stageTransfers(stages.front())) {
    if (transfer.isSourceMemRef() &&
        !isScratchpad(transfer.getSource(), memory_tree) &&
        isScratchpad(transfer.getDestination(), memory_tree)) {
      load = transfer;
      break;
    }
  }
  if (!load) return std::optional<Restick>();
  auto x = layoutOf(load.getSource());
  if (mlir::failed(x) || x->strides.empty() || x->strides.back() == 1)
    return std::optional<Restick>();  // already moves contiguous sticks

  // From here on this is a re-stick: every unmet expectation is an error.
  auto fail = [](mlir::Operation* at, const llvm::Twine& msg) {
    mlir::InFlightDiagnostic diag = at->emitError(PASS_NAME ": ");
    diag << msg;
    return diag;
  };

  std::optional<int64_t> p = unitStrideDim(*x);
  if (!p) return fail(load, "the source view has no dim with stride 1");
  int64_t rank = x->sizes.size();

  mlir::ktdf::DataTransferOp store;
  for (auto transfer : stageTransfers(stages.back())) {
    if (transfer.isDestMemRef() &&
        !isScratchpad(transfer.getDestination(), memory_tree) &&
        isScratchpad(transfer.getSource(), memory_tree)) {
      store = transfer;
      break;
    }
  }
  if (!store)
    return fail(load, "no transfer in the last stage stores to a global view");
  auto y = layoutOf(store.getDestination());
  if (mlir::failed(y) || !isRowMajor(*y))
    return fail(store, "the destination view is not row-major");
  if ((int64_t)y->sizes.size() != rank)
    return fail(store, "the source and destination views differ in rank (")
           << rank << " vs " << y->sizes.size() << ")";

  // Only fp16 sticks have been run on the device; other element sizes need
  // their own block size and transfer modes checked first.
  auto element_bytes = scheduler::tryGetSizeInBytes(
      mlir::cast<mlir::MemRefType>(load.getSource().getType())
          .getElementType());
  if (!element_bytes || *element_bytes != 2)
    return fail(load, "only 2-byte elements are supported, got ")
           << (element_bytes ? (int64_t)*element_bytes : -1) << " bytes";

  // The stick: what the load already moves along x's last dim.
  auto load_sizes = load.getMixedSourceSizes();
  std::optional<int64_t> stick = staticSize(load_sizes.back());
  if (!stick || *stick <= 1)
    return fail(load, "the load's last size is not a static stick length");
  int64_t block = *stick;

  int64_t extent = x->sizes[*p];
  if (extent % block != 0)
    return fail(load, "dim ") << *p << " has extent " << extent
                              << ", not a multiple of the stick size "
                              << block;

  // Each transfer moves one stick: size `block` in the last slot, 1 elsewhere.
  for (auto [transfer, sizes] :
       {std::pair{load, load.getMixedSourceSizes()},
        std::pair{store, store.getMixedDestSizes()}}) {
    for (auto [slot, size] : llvm::enumerate(sizes)) {
      int64_t want = (int64_t)slot == rank - 1 ? block : 1;
      if (staticSize(size) != want)
        return fail(transfer, "global-side size in slot ")
               << slot << " is not " << want
               << "; the pass expects one stick per transfer";
    }
  }

  // The load and the store must address the same element in every slot:
  // only then is the tile read from x the tile written to y. Each stage runs
  // its own loop over a staging-buffer dim; pair the load's loop over buffer
  // dim k with the store's.
  int64_t lead = load.getDestMap()->getNumResults() - rank;
  if (lead < 0 || (int64_t)store.getSourceMap()->getNumResults() - rank != lead)
    return fail(load,
                "the staging buffers must have the view rank plus the "
                "same number of loop dims");
  llvm::DenseMap<mlir::Value, mlir::Value> paired_ivs;
  for (int64_t k = 0; k < lead; ++k) {
    mlir::Value load_iv = indexAt(*load.getDestMap(), load.getDestIndices(), k);
    mlir::Value store_iv =
        indexAt(*store.getSourceMap(), store.getSourceIndices(), k);
    if (load_iv && store_iv && load_iv != store_iv &&
        mlir::scf::getForInductionVarOwner(load_iv) &&
        mlir::scf::getForInductionVarOwner(store_iv))
      paired_ivs[load_iv] = store_iv;
  }
  mlir::AffineMap load_map = *load.getSourceMap();
  mlir::AffineMap store_map = *store.getDestMap();
  for (int64_t slot = 0; slot < rank; ++slot) {
    if (!sameIndexExpr(load_map.getResult(slot), load.getSourceIndices(),
                       store_map.getResult(slot), store.getDestIndices(),
                       paired_ivs))
      return fail(store, "slot ")
             << slot << " of the store to y does not use the same index as "
             << "slot " << slot << " of the load from x; a re-stick needs "
             << "the load and the store to address the same element in "
             << "every slot";
  }

  mlir::ktdf::PipelineOp nested;
  for (auto stage : stages)
    stage->walk([&](mlir::ktdf::PipelineOp inner) { nested = inner; });
  if (!nested)
    return fail(pipeline, "no nested pipeline moves sticks from A to B");

  Restick plan{load, store, nested, *p, rank, block, {}, {}};

  // The loop over dim p, from what indexes slot p of the load.
  mlir::Value slot_p = indexAt(load.getSourceMap().value(),
                               load.getSourceIndices(), *p);
  if (slot_p && mlir::scf::getForInductionVarOwner(slot_p)) {
    // Not tiled: the slot is the loop's own IV, and the loop walks the whole
    // dim, so the dim must be one stick.
    if (extent != block)
      return fail(load, "dim ")
             << *p << " is walked by an untiled loop over " << extent
             << " elements; only one stick (" << block << ") is supported";
    plan.point_iv = slot_p;
  } else if (auto lin = slot_p ? slot_p.getDefiningOp<
                                     mlir::ktdf::TilingLinearizeIndexOp>()
                               : nullptr;
             lin && lin.getIvs().size() == 2) {
    plan.point_iv = lin.getIvs()[1];
    plan.tile =
        lin.getStrides()[0].getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
    if (!plan.tile)
      return fail(load, "the tile size of dim ")
             << *p << " is not a ktdf.tiling.reserve_size placeholder";
  } else {
    return fail(load, "slot ")
           << *p << " of the load is not indexed by a loop over dim " << *p
           << " (expected a loop IV or `linearize_index [tile_iv : tile], "
              "[point_iv : 1]`)";
  }
  if (!mlir::scf::getForInductionVarOwner(plan.point_iv))
    return fail(load, "the point index of dim ") << *p << " is not a loop IV";
  return std::optional<Restick>(plan);
}

//===----------------------------------------------------------------------===//
// Widening a re-stick
//===----------------------------------------------------------------------===//

/// The backing memref.alloc of a staging buffer, through the ktdf.private
/// result it is yielded as.
struct StagingBuffer {
  mlir::Value value;  // the ktdf.private result the transfers use
  mlir::ktdf::PrivateOp private_op;
  unsigned result = 0;
  mlir::memref::AllocOp alloc;
};

mlir::FailureOr<StagingBuffer> stagingBufferOf(mlir::Value value) {
  StagingBuffer buffer;
  buffer.value = value;
  buffer.private_op = value.getDefiningOp<mlir::ktdf::PrivateOp>();
  if (!buffer.private_op) return mlir::failure();
  buffer.result = mlir::cast<mlir::OpResult>(value).getResultNumber();
  buffer.alloc = buffer.private_op.getYieldOp()
                     .getOperand(buffer.result)
                     .getDefiningOp<mlir::memref::AllocOp>();
  if (!buffer.alloc) return mlir::failure();
  return buffer;
}

/// Gives each dim in `new_sizes` of `buffer` that static size, dropping the
/// dynamic size operand of any dim that had one.
void resizeBuffer(StagingBuffer& buffer,
                  llvm::ArrayRef<std::pair<int64_t, int64_t>> new_sizes,
                  mlir::OpBuilder& builder) {
  mlir::MemRefType type = buffer.alloc.getType();
  llvm::SmallVector<int64_t> shape(type.getShape());
  for (auto [dim, size] : new_sizes) shape[dim] = size;
  llvm::SmallVector<mlir::Value> dynamic_sizes;
  unsigned next_dynamic = 0;
  for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
    if (!mlir::ShapedType::isDynamic(extent)) continue;
    mlir::Value operand = buffer.alloc.getDynamicSizes()[next_dynamic++];
    if (mlir::ShapedType::isDynamic(shape[dim]))
      dynamic_sizes.push_back(operand);
  }
  auto new_type = mlir::MemRefType::get(shape, type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface{},
                                        type.getMemorySpace());
  builder.setInsertionPoint(buffer.alloc);
  auto new_alloc = mlir::memref::AllocOp::create(
      builder, buffer.alloc.getLoc(), new_type, dynamic_sizes);
  buffer.alloc.getResult().replaceAllUsesWith(new_alloc.getResult());
  buffer.alloc.erase();
  buffer.alloc = new_alloc;
  buffer.private_op.getResult(buffer.result).setType(new_type);
}

/// Builds x_t from `x`, the load's source: the same memory as x with dims `p`
/// and `last` swapped in sizes, strides and coordinate set, so x_t's last dim
/// has stride 1. x's chain of view and casts is cloned, each clone right after
/// its original; x and its users are not changed. Element [.., v, .., u] of
/// x_t (v in slot p, u in slot last) is element [.., u, .., v] of x.
mlir::FailureOr<mlir::Value> buildSwappedView(mlir::Value x, int64_t p,
                                              int64_t last,
                                              mlir::OpBuilder& builder) {
  llvm::SmallVector<int64_t> order(last + 1);
  std::iota(order.begin(), order.end(), 0);
  std::swap(order[p], order[last]);

  mlir::Operation* def = x.getDefiningOp();
  auto type = mlir::dyn_cast<mlir::MemRefType>(x.getType());
  auto new_type = type ? permutedMemRefType(type, order)
                       : mlir::FailureOr<mlir::MemRefType>(mlir::failure());
  if (mlir::failed(new_type))
    return (def ? def : x.getParentBlock()->getParentOp())
        ->emitError(PASS_NAME
                    ": the source view has no strided "
                    "memref type");

  if (auto view =
          mlir::dyn_cast_or_null<mlir::ktdp::ConstructMemoryViewOp>(def)) {
    builder.setInsertionPointAfter(view);
    auto x_t =
        mlir::cast<mlir::ktdp::ConstructMemoryViewOp>(builder.clone(*view));
    permuteViewOp(x_t, order, *new_type);
    return x_t.getResult();
  }
  if (mlir::isa_and_nonnull<mlir::memref::CastOp,
                            mlir::memref::MemorySpaceCastOp>(def)) {
    auto source = buildSwappedView(def->getOperand(0), p, last, builder);
    if (mlir::failed(source)) return mlir::failure();
    mlir::IRMapping mapping;
    mapping.map(def->getOperand(0), *source);
    builder.setInsertionPointAfter(def);
    mlir::Operation* clone = builder.clone(*def, mapping);
    clone->getResult(0).setType(*new_type);
    return clone->getResult(0);
  }
  return (def ? def : x.getParentBlock()->getParentOp())
      ->emitError(PASS_NAME
                  ": the source view is not a "
                  "ktdp.construct_memory_view reached "
                  "through memref.cast or "
                  "memref.memory_space_cast");
}

/// Erases `value`'s defining view or cast once nothing uses it, then the one
/// it was cast from, and so on.
void eraseUnusedViewChain(mlir::Value value) {
  while (mlir::Operation* def = value.getDefiningOp()) {
    if (!def->use_empty() ||
        !mlir::isa<mlir::ktdp::ConstructMemoryViewOp, mlir::memref::CastOp,
                   mlir::memref::MemorySpaceCastOp>(def))
      return;
    mlir::Value next = mlir::isa<mlir::ktdp::ConstructMemoryViewOp>(def)
                           ? mlir::Value()
                           : def->getOperand(0);
    def->erase();
    if (!next) return;
    value = next;
  }
}

/// Widens the re-stick `plan` to `block` x `block` tiles. See the file
/// comment for the shape of the result.
mlir::LogicalResult widenRestick(Restick plan, mlir::OpBuilder& builder) {
  mlir::MLIRContext* ctx = plan.load.getContext();
  const int64_t p = plan.p, last = plan.rank - 1, block = plan.block;
  mlir::Location loc = plan.load.getLoc();

  // The constants every step uses, once each, at the top of the function so
  // they dominate every loop of the kernel.
  mlir::Block* entry =
      &plan.load->getParentOfType<mlir::func::FuncOp>().getBody().front();
  mlir::Value c0 = indexConstant(entry, 0, builder);
  mlir::Value c1 = indexConstant(entry, 1, builder);
  mlir::Value c_block = indexConstant(entry, block, builder);

  auto a = stagingBufferOf(plan.load.getDestination());
  auto b = stagingBufferOf(plan.store.getSource());
  if (mlir::failed(a) || mlir::failed(b))
    return plan.load->emitError(PASS_NAME
                                ": a staging buffer is not a ktdf.private "
                                "alloc");
  // A buffer index lists the stage's point loops first, then one dim per view
  // dim: view dim d is buffer dim lead + d.
  const int64_t lead = a->alloc.getType().getRank() - plan.rank;
  if (lead < 0 || b->alloc.getType().getRank() - plan.rank != lead)
    return plan.load->emitError(PASS_NAME
                                ": staging buffers must have the view rank "
                                "plus the same number of loop dims");

  // The buffer dim that holds the point loop over dim p, if the point loop
  // is inside the pipeline. Every stage indexes it with its own copy of that
  // loop.
  std::optional<int64_t> kp;
  for (int64_t k = 0; k < lead; ++k)
    if (indexAt(plan.load.getDestMap().value(), plan.load.getDestIndices(),
                k) == plan.point_iv)
      kp = k;

  // 1. Every point loop over dim p runs once: the widened transfers cover the
  //    whole stick in one go. The loops are the one indexing slot p of the
  //    load and, in each stage, the one indexing buffer dim kp.
  llvm::SetVector<mlir::scf::ForOp> point_loops;
  point_loops.insert(mlir::scf::getForInductionVarOwner(plan.point_iv));
  if (kp) {
    llvm::SmallVector<mlir::ktdf::DataTransferOp> on_buffers;
    plan.load->getParentOfType<mlir::ktdf::PipelineOp>()->walk(
        [&](mlir::ktdf::DataTransferOp op) { on_buffers.push_back(op); });
    for (auto op : on_buffers) {
      for (bool source : {true, false}) {
        mlir::Value side = source ? op.getSource() : op.getDestination();
        if (side != a->value && side != b->value) continue;
        auto map = source ? op.getSourceMap() : op.getDestMap();
        mlir::Value iv =
            indexAt(*map, source ? op.getSourceIndices() : op.getDestIndices(),
                    *kp);
        auto loop = iv ? mlir::scf::getForInductionVarOwner(iv)
                       : mlir::scf::ForOp();
        if (!loop)
          return op->emitError(PASS_NAME ": staging-buffer dim ")
                 << *kp << " is not indexed by a loop";
        point_loops.insert(loop);
      }
    }
  }

  // 2. A tiled loop over dim p gets one stick per tile: the placeholder
  //    becomes `block`. Every op stage coarsening derived from it (the tile
  //    loop's trip count, the linearize_index strides, the buffers' sizes)
  //    follows.
  if (plan.tile) {
    plan.tile.replaceAllUsesWith(c_block);
    plan.tile.erase();
  }
  for (mlir::scf::ForOp loop : point_loops) loop.setUpperBound(c1);

  // 3. The buffers hold one tile per point of the remaining loops: view dim p
  //    grows to `block`, the point dim over p shrinks to 1.
  llvm::SmallVector<std::pair<int64_t, int64_t>> new_sizes = {
      {lead + p, block}};
  if (kp) new_sizes.push_back({*kp, 1});
  resizeBuffer(*a, new_sizes, builder);
  resizeBuffer(*b, new_sizes, builder);

  // 4. The load and store move a whole tile: size `block` in slot p as well
  //    as in the last slot, on both sides.
  //
  //    Lowering moves a transfer's last slot fastest, as the 64 lanes of one
  //    vector, on each side. Row a of A (buffer dim lead + p = a) must then
  //    hold stick a of x, the 64 contiguous elements x[.., p = 0..63, ..,
  //    last = a]. So the load reads from x_t, whose last slot is x's dim p,
  //    with its slots p and last swapped to match:
  //
  //      A[lead + p = a, last = b]  <-  x_t[p = a, last = b]
  //                                  =  x[p = b, last = a]
  //                                  =  X0 + stride_x[last] * a + b
  //
  //    where X0 is the address of the tile's first element. Reading x itself
  //    would fill row a with element a of every stick (X0 + a +
  //    stride_x[last] * b): a transpose of its own, which the splat/extract
  //    below would undo.
  mlir::Value x = plan.load.getSource();
  auto x_t = buildSwappedView(x, p, last, builder);
  if (mlir::failed(x_t)) return mlir::failure();
  for (auto [op, global_is_source] :
       {std::pair{plan.load, true}, std::pair{plan.store, false}}) {
    TransferParts parts(op);
    (global_is_source ? parts.src_sizes : parts.dst_sizes)[p] =
        indexAttr(ctx, block);
    (global_is_source ? parts.dst_sizes : parts.src_sizes)[lead + p] =
        indexAttr(ctx, block);
    if (global_is_source) {
      parts.source = *x_t;
      llvm::SmallVector<mlir::AffineExpr> results(parts.src_map.getResults());
      std::swap(results[p], results[last]);
      parts.src_map =
          mlir::AffineMap::get(parts.src_map.getNumDims(), 0, results, ctx);
      std::swap(parts.src_sizes[p], parts.src_sizes[last]);
    }
    parts.rebuild(op, builder);
  }
  eraseUnusedViewChain(x);

  // 5. The nested pipeline transposes each tile one element at a time:
  //    element (r, c) of A goes to (c, r) of B. Wrap it in loops r, c over
  //    [0, block). Its read from A becomes a one-element splat at
  //    (lead + p, last) = (r, c); its write to B a one-element extract at
  //    (c, r).
  builder.setInsertionPoint(plan.nested);
  auto r_loop = mlir::scf::ForOp::create(builder, loc, c0, c_block, c1);
  builder.setInsertionPoint(r_loop.getBody()->getTerminator());
  auto c_loop = mlir::scf::ForOp::create(builder, loc, c0, c_block, c1);
  plan.nested->moveBefore(c_loop.getBody()->getTerminator());
  mlir::Value r = r_loop.getInductionVar(), c = c_loop.getInductionVar();

  for (mlir::ktdf::StageOp stage : plan.nested.getStages()) {
    for (mlir::ktdf::DataTransferOp op : stageTransfers(stage)) {
      bool reads_buffer = op.getSource() == a->value;
      bool writes_buffer = op.getDestination() == b->value;
      if (!reads_buffer && !writes_buffer) continue;
      TransferParts parts(op);
      auto& map = reads_buffer ? parts.src_map : parts.dst_map;
      auto& indices = reads_buffer ? parts.src_indices : parts.dst_indices;
      auto& sizes = reads_buffer ? parts.src_sizes : parts.dst_sizes;
      // One index per buffer dim through an identity map: the point-loop
      // indices stay, every view dim is 0 except the two the element is at.
      llvm::SmallVector<mlir::Value> element(lead + plan.rank, c0);
      for (int64_t k = 0; k < lead; ++k) {
        element[k] = indexAt(map, indices, k);
        if (!element[k])
          return op->emitError(PASS_NAME ": staging-buffer dim ")
                 << k << " is not indexed by a loop";
      }
      element[lead + p] = reads_buffer ? r : c;
      element[lead + last] = reads_buffer ? c : r;
      map = mlir::AffineMap::getMultiDimIdentityMap(element.size(), ctx);
      indices = element;
      llvm::fill(sizes, indexAttr(ctx, 1));
      parts.rebuild(op, builder)
          ->setDiscardableAttr(
              "transfer_mode",
              mlir::StringAttr::get(ctx, reads_buffer ? "splat" : "extract"));
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Listing views by stride
//===----------------------------------------------------------------------===//

/// Rewrites `view` to list its dims in `order`, and every transfer on it to
/// list its map results and sizes the same way. Each dim keeps its size and
/// stride, so every element keeps its address.
mlir::LogicalResult reorderView(mlir::ktdp::ConstructMemoryViewOp view,
                                llvm::ArrayRef<int64_t> order,
                                mlir::OpBuilder& builder) {
  mlir::MLIRContext* ctx = view.getContext();
  auto new_type = permutedMemRefType(
      mlir::cast<mlir::MemRefType>(view.getResult().getType()), order);
  if (mlir::failed(new_type))
    return view->emitError(PASS_NAME ": the view has no strided memref type");
  permuteViewOp(view, order, *new_type);

  // Carry the type through the casts, and collect the transfers on the view.
  llvm::SmallVector<std::pair<mlir::ktdf::DataTransferOp, bool>> transfers;
  llvm::SmallVector<mlir::Value> worklist = {view.getResult()};
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto value_type = mlir::cast<mlir::MemRefType>(value.getType());
    for (mlir::Operation* user : value.getUsers()) {
      if (mlir::isa<mlir::memref::CastOp, mlir::memref::MemorySpaceCastOp>(
              user)) {
        auto result_type =
            mlir::cast<mlir::MemRefType>(user->getResult(0).getType());
        user->getResult(0).setType(mlir::MemRefType::get(
            value_type.getShape(), value_type.getElementType(),
            value_type.getLayout(), result_type.getMemorySpace()));
        worklist.push_back(user->getResult(0));
      } else if (auto transfer =
                     mlir::dyn_cast<mlir::ktdf::DataTransferOp>(user)) {
        transfers.push_back({transfer, transfer.getSource() == value});
      } else {
        return user->emitError(PASS_NAME
                               ": a view listed by stride is used by an op "
                               "other than a cast or a ktdf.data_transfer");
      }
    }
  }

  // Each transfer on the view lists its map results and sizes in the same
  // order as the view's dims.
  for (auto [transfer, is_source] : transfers) {
    TransferParts parts(transfer);
    auto& map = is_source ? parts.src_map : parts.dst_map;
    auto& sizes = is_source ? parts.src_sizes : parts.dst_sizes;
    map = mlir::AffineMap::get(map.getNumDims(), 0,
                               permuted(map.getResults(), order), ctx);
    sizes = permuted(llvm::ArrayRef(sizes), order);
    parts.rebuild(transfer, builder);
  }
  return mlir::success();
}

/// Lists every global view of `root` by decreasing stride.
mlir::LogicalResult orderViewsByStride(
    mlir::Operation* root,
    const scheduler::arch_view::MemoryTree& memory_tree,
    mlir::OpBuilder& builder) {
  llvm::SmallVector<mlir::ktdp::ConstructMemoryViewOp> views;
  root->walk([&](mlir::ktdp::ConstructMemoryViewOp view) {
    if (!isScratchpad(view.getResult(), memory_tree)) views.push_back(view);
  });
  for (auto view : views) {
    if (llvm::any_of(view.getStaticSizes(), mlir::ShapedType::isDynamic) ||
        llvm::any_of(view.getStaticStrides(), mlir::ShapedType::isDynamic))
      continue;  // only static views are described by their strides
    if (stridesDecrease(view.getStaticSizes(), view.getStaticStrides()))
      continue;
    auto order = strideOrder(view);
    if (mlir::failed(reorderView(view, order, builder))) return mlir::failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DataTransferAlignmentPass
    : public scheduler::impl::DataTransferAlignmentPassBase<
          DataTransferAlignmentPass> {
  void runOnOperation() override {
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* device = device_manager.getOrImportDevice();
    if (!device) {
      getOperation()->emitError(PASS_NAME ": failed to import device spec");
      return signalPassFailure();
    }
    auto& memory_tree =
        device_manager.getOrCreateView<scheduler::arch_view::MemoryTree>(
            *device);

    // Outermost pipelines only: a re-stick's nested pipeline is part of it.
    llvm::SmallVector<mlir::ktdf::PipelineOp> pipelines;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::ktdf::PipelineOp pipeline) {
          pipelines.push_back(pipeline);
          return mlir::WalkResult::skip();
        });

    mlir::OpBuilder builder(&getContext());
    for (auto pipeline : pipelines) {
      auto plan = findRestick(pipeline, memory_tree);
      if (mlir::failed(plan)) return signalPassFailure();
      if (!*plan) continue;
      LDBG(1) << "re-stick: p = " << (*plan)->p << ", block "
              << (*plan)->block << " at " << pipeline.getLoc();
      if (mlir::failed(widenRestick(**plan, builder)))
        return signalPassFailure();
    }

    if (orderViews &&
        mlir::failed(orderViewsByStride(getOperation(), memory_tree, builder)))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createDataTransferAlignmentPass() {
  return std::make_unique<DataTransferAlignmentPass>();
}
