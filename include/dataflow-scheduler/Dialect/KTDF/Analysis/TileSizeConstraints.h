//===-- TileSizeConstraints.h -----------------------------------*- c++ -*-===//
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
// The constraints on a ktdf.tiling.reserve_size, and the tile sizes they
// allow.
//
// Passes that need something from a tile size state it on its reserve_size
// (min_value, divisibility); the loops it tiles add that the tile must divide
// each loop's total. This analysis is the one definition of which tile sizes
// those constraints allow, shared by the pass that chooses a tile size and by
// passes that must reason about it before it is chosen.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_TILESIZECONSTRAINTS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_TILESIZECONSTRAINTS_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <cstdint>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// A loop tiled by a tile size, recognized from
///   %bound = arith.ceildivui %total_size, %tile_size
///   scf.for %i = %c0 to %bound step %c1
struct TiledLoop {
  scf::ForOp loop;
  /// The total size being tiled, which the tile size must divide.
  int64_t total_size;
};

/// The constraints on one ktdf.tiling.reserve_size: its own min_value and
/// divisibility, and the totals of the loops it tiles.
class TileSizeConstraints {
 public:
  /// Collects the constraints on `reserve_size_op` from its attributes and
  /// uses.
  explicit TileSizeConstraints(TilingReserveSizeOp reserve_size_op);

  [[nodiscard]] auto getReserveSizeOp() const -> TilingReserveSizeOp {
    return reserve_size_op_;
  }
  [[nodiscard]] auto getMinValue() const -> int64_t { return min_value_; }
  [[nodiscard]] auto getDivisibility() const -> int64_t {
    return divisibility_;
  }
  [[nodiscard]] auto getTiledLoops() const -> llvm::ArrayRef<TiledLoop> {
    return tiled_loops_;
  }

  /// Uses of the tile size that do not fit the tiled-loop pattern, and so do
  /// not constrain it; one reason per use, for diagnostics.
  [[nodiscard]] auto getUnrecognizedUses() const
      -> llvm::ArrayRef<llvm::StringRef> {
    return unrecognized_uses_;
  }

  /// Returns true if the tile size tiles at least one loop. Without one there
  /// is no total to divide, and so nothing to choose a tile size against.
  [[nodiscard]] auto hasTiledLoops() const -> bool {
    return !tiled_loops_.empty();
  }

  /// Returns true if `tile_size` satisfies every constraint, with the
  /// divisibility further raised to a multiple of `extra_divisibility`.
  [[nodiscard]] auto isLegal(int64_t tile_size,
                             int64_t extra_divisibility = 1) const -> bool;

  /// Returns every legal tile size (see isLegal), ascending. A legal tile size
  /// divides every non-zero total, so the smallest non-zero total bounds the
  /// set. A loop with no iterations does not constrain the tile size; if no
  /// loop has iterations, every multiple of the divisibility from min_value up
  /// is legal and only the smallest is returned.
  [[nodiscard]] auto legalTileSizes(int64_t extra_divisibility = 1) const
      -> llvm::SmallVector<int64_t>;

  /// Returns true if some tile size is legal.
  [[nodiscard]] auto isFeasible(int64_t extra_divisibility = 1) const -> bool {
    return !legalTileSizes(extra_divisibility).empty();
  }

 private:
  TilingReserveSizeOp reserve_size_op_;
  int64_t min_value_;
  int64_t divisibility_;
  llvm::SmallVector<TiledLoop> tiled_loops_;
  llvm::SmallVector<llvm::StringRef> unrecognized_uses_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_TILESIZECONSTRAINTS_H_
