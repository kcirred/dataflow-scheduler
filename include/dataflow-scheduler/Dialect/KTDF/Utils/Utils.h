//===-- Utils.h -------------------------------------------------*- c++ -*-===//
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
// This file declares utilities for the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_

#include <llvm/ADT/SmallVector.h>

#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ktdf {

/// Collect all stages nested under a root via walk
void collectStages(Operation* root, SmallVectorImpl<StageOp>& stages);

/// Collect all queried units (uniform.query_map results)
void collectQueriedUnits(Operation* root,
                         SmallVectorImpl<Operation*>& query_ops);

/// Check if a memref value is written to within the given operation's region.
/// Returns true if any ktdf.data_transfer operation writes to the memref as a
/// destination.
/// \param memref The memref value to check
/// \param root_op The root operation to search within
/// \return true if the memref is written to, false otherwise
auto isTransferTarget(Value memref, Operation* root_op) -> bool;

/// Find stage for a given queried unit
// FIXME: This function has no users.
auto findStageForUnit(uniform::QueryMapOp query_op,
                      ArrayRef<ktdf::StageOp> stages) -> StageOp;

/// Mark `transfer` as a splat: the hardware fills the destination by
/// replicating the source rather than by copying element for element.
void markSplatTransfer(ktdf::DataTransferOp transfer);

/// Whether `transfer` carries the splat mark `markSplatTransfer` sets.
auto isSplatTransfer(ktdf::DataTransferOp transfer) -> bool;

/// Whether `transfer` fills a destination wider than its source, either by
/// replicating the source (splat) or by zero-filling the remainder (pad).
auto isBroadcastTransfer(ktdf::DataTransferOp transfer) -> bool;

/// Whether `transfer` extracts a narrower slice out of a wider source, so the
/// destination is smaller than the data received.
auto isExtractTransfer(ktdf::DataTransferOp transfer) -> bool;

/// Overwrite the innermost `static_source_sizes` entry of `transfer` with
/// `size`.  Does nothing, and returns false, when `transfer` carries no static
/// source sizes.
auto setInnermostStaticSourceSize(ktdf::DataTransferOp transfer, int64_t size)
    -> bool;

/// Convert all uses of a tensor SSA value to memref semantics throughout the
/// IR reachable from that tensor.
///
/// Starting from `tensor_val`, the function collects every op that directly or
/// transitively consumes a tensor-typed value derived from it, then converts
/// each one:
///
///   `linalg.generic`
///     Rebuilt in pure buffer-semantics via `cloneLinalgGenericAsBufferOp`.
///     Each tensor result is replaced (RAUW) with the corresponding output
///     memref.  The original op is erased.
///
///   `ktdf.write_to_fifo`
///     Data operand is reassigned in-place to the memref replacement.
///
///   `ktdf.read_from_fifo` (tensor result)
///     Re-emitted as a memref-typed read via `convertFromTensorToMemref`, so
///     downstream users see the memref.  The original op is erased.
///
///   Any other op
///     Returns `failure()` immediately; the IR is left in a partially-
///     converted state.  The caller is responsible for signalling pass failure.
///
/// The op that produced `tensor_val` is erased once nothing reads it.  Anything
/// else the conversion leaves dead is the caller's to clean up.
///
/// \param builder     OpBuilder used to insert new ops; its insertion point is
///                    managed internally.
/// \param tensor_val  The tensor SSA value to replace.  Must have
///                    `RankedTensorType`.
/// \param memref_val  The memref Value to substitute for `tensor_val`.  Must
///                    have a `MemRefType` with the same shape and element type.
/// \param extra_subst  Further tensor→memref pairs to substitute alongside
///                    `tensor_val`.  A converted op's other operands are
///                    looked up here, so a caller that has already provided a
///                    buffer for them -- a freshly allocated output, or a
///                    sibling input it re-emitted as a memref read -- does not
///                    see them passed through as tensors.
/// \return            `success()` if all reachable ops were converted;
///                    `failure()` if an unrecognised op was encountered.
auto convertTensorUsesToMemref(
    mlir::OpBuilder& builder, mlir::Value tensor_val, mlir::Value memref_val,
    const llvm::DenseMap<mlir::Value, mlir::Value>& extra_subst =
        llvm::DenseMap<mlir::Value, mlir::Value>()) -> mlir::LogicalResult;

/// Re-emit a tensor-typed `ktdf.read_from_fifo` as a memref-typed one.
///
/// Inserts a new `ktdf.read_from_fifo` immediately before `read_op` that
/// produces a plain (no layout, no memory space) `memref` of the same shape and
/// element type as the original tensor result.  The original op is NOT erased;
/// the caller is responsible for removing it once its uses are gone.
///
/// The new read takes the original's place rather than `builder`'s insertion
/// point: slots that share a physical FIFO are drained in program order, so
/// moving a read past a sibling's swaps the values the two receive.
///
/// A splat mode the original carries is carried over: reading the slot as a
/// buffer does not perform the shuffle it names, so the new read still owes it.
///
/// \param builder  OpBuilder to create with; its insertion point is restored.
/// \param read_op  The tensor-typed read_from_fifo to convert.
/// \return         The memref-typed result Value of the new read.
auto convertFromTensorToMemref(mlir::OpBuilder& builder,
                               ktdf::ReadFromFifoOp read_op) -> mlir::Value;

/// Clone the body and attributes of `generic_op` into a new buffer-semantics
/// `linalg.generic` that operates entirely on memrefs.
///
/// The new op has no SSA results (buffer mode: writes happen in-place into
/// `outputs`).  The body region of `generic_op` is cloned verbatim; the
/// placeholder block that `cloneInto` prepends is dropped automatically.
///
/// The new op is inserted at `builder`'s insertion point; `generic_op` is NOT
/// erased.
///
/// \param builder   OpBuilder positioned at the desired insertion point.
/// \param generic_op  The tensor-semantic source op (must have pure-tensor or
///                    mixed semantics — its body is cloned, not the operands).
/// \param inputs    Memref Values to use as `ins` operands.
/// \param outputs   Memref Values to use as `outs` operands.
/// \return          The newly created buffer-mode `linalg.generic`.
auto cloneLinalgGenericAsBufferOp(mlir::OpBuilder& builder,
                                  mlir::linalg::GenericOp generic_op,
                                  mlir::ValueRange inputs,
                                  mlir::ValueRange outputs)
    -> mlir::linalg::GenericOp;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_
