//===-- Utils.cpp -----------------------------------------------*- c++ -*-===//
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
// This file implements utilities for the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

#include <utility>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "ktdf-utilities"

using namespace mlir;
using namespace mlir::ktdf;

void mlir::ktdf::collectStages(Operation* root,
                               SmallVectorImpl<StageOp>& stages) {
  root->walk([&](StageOp stage) { stages.push_back(stage); });
}

void mlir::ktdf::collectQueriedUnits(Operation* root,
                                     SmallVectorImpl<Operation*>& query_ops) {
  root->walk([&](uniform::QueryMapOp op) { query_ops.push_back(op); });
}

auto mlir::ktdf::isTransferTarget(Value memref, Operation* root_op) -> bool {
  if (!root_op) {
    return false;
  }

  return root_op
      ->walk([&](DataTransferOp transfer) {
        // Check if memref is the destination
        if (transfer.getDestination() == memref) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

namespace {

auto isUsedInRegion(Value value, Region& region) -> bool {
  return llvm::any_of(value.getUsers(), [&](Operation* user) -> bool {
    return region.isAncestor(user->getParentRegion());
  });
}

}  // namespace

namespace {

/// Name of the discardable attribute recording how a data transfer fills its
/// destination, and the modes that widen it.
constexpr StringLiteral kTransferModeAttrName = "transfer_mode";
constexpr StringLiteral kSplatTransferMode = "splat";
constexpr StringLiteral kPadTransferMode = "pad";
constexpr StringLiteral kExtractTransferMode = "extract";

StringRef getTransferMode(ktdf::DataTransferOp transfer) {
  auto mode = llvm::dyn_cast_if_present<StringAttr>(
      transfer->getDiscardableAttr(kTransferModeAttrName));
  return mode ? mode.getValue() : StringRef();
}

}  // namespace

void mlir::ktdf::markSplatTransfer(ktdf::DataTransferOp transfer) {
  transfer->setDiscardableAttr(
      kTransferModeAttrName,
      StringAttr::get(transfer->getContext(), kSplatTransferMode));
}

auto mlir::ktdf::isSplatTransfer(ktdf::DataTransferOp transfer) -> bool {
  return getTransferMode(transfer) == kSplatTransferMode;
}

auto mlir::ktdf::isBroadcastTransfer(ktdf::DataTransferOp transfer) -> bool {
  StringRef mode = getTransferMode(transfer);
  return mode == kSplatTransferMode || mode == kPadTransferMode;
}

auto mlir::ktdf::isExtractTransfer(ktdf::DataTransferOp transfer) -> bool {
  return getTransferMode(transfer) == kExtractTransferMode;
}

auto mlir::ktdf::setInnermostStaticSourceSize(ktdf::DataTransferOp transfer,
                                              int64_t size) -> bool {
  auto sizes = transfer.getStaticSourceSizesArray();
  if (!sizes || sizes->empty()) return false;

  SmallVector<int64_t> new_sizes(*sizes);
  new_sizes.back() = size;
  transfer.setStaticSourceSizes(ArrayRef<int64_t>(new_sizes));
  return true;
}

// ---------------------------------------------------------------------------
// Collect every op reachable from `seed` through tensor-typed values.
//
// Starting at the direct users of `seed`, each user's tensor-typed results are
// followed to their own users, and so on.  Insertion order puts a def before
// the users it reaches, which is the order the conversion below needs: an op is
// rewritten only once the buffers replacing its tensor operands exist.
// ---------------------------------------------------------------------------
static void collectAffectedOps(Value seed,
                               llvm::SetVector<Operation*>& affected) {
  // Worklist of tensor Values whose users we still need to visit.
  llvm::SmallVector<Value> worklist{seed};
  llvm::DenseSet<Value> visited{seed};

  while (!worklist.empty()) {
    Value tensor = worklist.pop_back_val();
    for (Operation* user : tensor.getUsers()) {
      affected.insert(user);
      for (Value result : user->getResults())
        if (isa<RankedTensorType>(result.getType()) &&
            visited.insert(result).second)
          worklist.push_back(result);
    }
  }
}

auto mlir::ktdf::convertTensorUsesToMemref(
    OpBuilder& builder, Value tensor_val, Value memref_val,
    const llvm::DenseMap<Value, Value>& extra_subst) -> LogicalResult {
  assert(isa<RankedTensorType>(tensor_val.getType()) &&
         "convertTensorUsesToMemref: tensor_val must have RankedTensorType");
  assert(isa<MemRefType>(memref_val.getType()) &&
         "convertTensorUsesToMemref: memref_val must have MemRefType");

  // Tensor Value → the memref standing in for it.  It grows as ops are
  // converted, so an op's operands are looked up once its producers are done.
  llvm::DenseMap<Value, Value> subst(extra_subst);
  subst[tensor_val] = memref_val;
  auto lookup = [&subst](Value v) -> Value {
    auto it = subst.find(v);
    return it != subst.end() ? it->second : v;
  };

  llvm::SetVector<Operation*> affected;
  collectAffectedOps(tensor_val, affected);

  for (Operation* op : affected) {
    // ── linalg.generic ──────────────────────────────────────────────────────
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      builder.setInsertionPoint(generic);

      SmallVector<Value> new_ins, new_outs;
      for (Value v : generic.getInputs()) new_ins.push_back(lookup(v));
      for (Value v : generic.getOutputs()) new_outs.push_back(lookup(v));

      cloneLinalgGenericAsBufferOp(builder, generic, new_ins, new_outs);

      // The buffer semantics op writes in place, so whatever read a result now
      // reads the matching output buffer.
      for (auto [result, out_memref] :
           llvm::zip(generic.getResults(), new_outs)) {
        subst[result] = out_memref;
        result.replaceAllUsesWith(out_memref);
      }
      op->erase();
      continue;
    }

    // ── ktdf.write_to_fifo ──────────────────────────────────────────────────
    if (auto write = dyn_cast<ktdf::WriteToFifoOp>(op)) {
      write.getDataMutable().assign(lookup(write.getData()));
      continue;
    }

    // ── ktdf.read_from_fifo ─────────────────────────────────────────────────
    if (auto read = dyn_cast<ktdf::ReadFromFifoOp>(op)) {
      // A memref result is already what the rebuilt consumers want.
      if (!isa<RankedTensorType>(read.getResult().getType())) continue;

      builder.setInsertionPoint(read);
      Value new_read = convertFromTensorToMemref(builder, read);
      subst[read.getResult()] = new_read;
      read.getResult().replaceAllUsesWith(new_read);
      op->erase();
      continue;
    }

    // ── Unknown op ──────────────────────────────────────────────────────────
    return op->emitError(
        "convertTensorUsesToMemref: cannot convert op from tensor to memref "
        "semantics — unrecognised op type");
  }

  // The op that produced the tensor has nothing reading it now.
  if (Operation* producer = tensor_val.getDefiningOp())
    if (producer->use_empty()) producer->erase();

  return success();
}

auto mlir::ktdf::convertFromTensorToMemref(OpBuilder& builder,
                                           ktdf::ReadFromFifoOp read_op)
    -> Value {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(read_op);
  auto tensor_type = cast<RankedTensorType>(read_op.getResult().getType());
  auto memref_type =
      MemRefType::get(tensor_type.getShape(), tensor_type.getElementType());
  return ktdf::ReadFromFifoOp::create(builder, read_op.getLoc(), memref_type,
                                      read_op.getFifoSlot(),
                                      read_op.getSplatAttr())
      .getResult();
}

auto mlir::ktdf::cloneLinalgGenericAsBufferOp(OpBuilder& builder,
                                              linalg::GenericOp generic_op,
                                              ValueRange inputs,
                                              ValueRange outputs)
    -> linalg::GenericOp {
  auto buf_generic = linalg::GenericOp::create(
      builder, generic_op.getLoc(),
      /*resultTensorTypes=*/TypeRange{}, inputs, outputs,
      generic_op.getIndexingMapsAttr(), generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it, keep the clone.
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();
  return buf_generic;
}

auto mlir::ktdf::findStageForUnit(uniform::QueryMapOp query_op,
                                  ArrayRef<StageOp> stages) -> StageOp {
  auto unit = query_op->getResult(0);

  for (auto stage : stages) {
    if (isUsedInRegion(unit, stage.getBodyRegion())) {
      return stage;
    }
  }

  return {};
}
