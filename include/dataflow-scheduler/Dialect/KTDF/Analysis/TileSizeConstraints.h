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
// each loop's total; and a loop distributed across instances (ktdf.parallel)
// adds that its trip count, built from the tile size, must be a multiple of the
// number of instances. This analysis is the one definition of which tile sizes
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
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// A loop tiled by a tile size, recognized from
///   %bound = arith.ceildivui %total_size, %tile_size
///   scf.for %i = %c0 to %bound step %c1
/// or a ktdf.parallel with that upper bound (the same loop, distributed).
struct TiledLoop {
  Operation* loop;
  /// The total size being tiled, which the tile size must divide.
  int64_t total_size;
};

/// A requirement on a tile size beyond those of its reserve_size and tiled
/// loops: it must be a multiple of `multiple_of` and, when `divides` is
/// non-zero, divide `divides`.
struct TileSizeRequirement {
  int64_t multiple_of = 1;
  int64_t divides = 0;

  /// Returns the requirement that satisfies both this one and `other`.
  [[nodiscard]] auto combine(const TileSizeRequirement& other) const
      -> TileSizeRequirement;
};

/// What distributing a loop across instances requires of the tile size its
/// trip count is built from.
struct SplitRequirement {
  TilingReserveSizeOp reserve_size_op;
  TileSizeRequirement requirement;
  /// No tile size can make the trip count a multiple of the number of
  /// instances (a tile loop whose total is not).
  bool impossible = false;
};

/// Returns what distributing a loop with upper bound `upper_bound` across
/// `num_instances` instances requires of the tile size that bound is built
/// from, so that every instance runs the same number of iterations:
///   - ktdf.tiling.derive_size [... : %tile] (a point loop): the trip count is
///     the tile size, so it must be a multiple of num_instances;
///   - arith.divui (that derive_size, F) for a constant F: the trip count is
///     tile / F, so the tile must be a multiple of F * num_instances;
///   - arith.ceildivui (total, %tile) (a tile loop): the trip count is
///     total / tile, so the tile must divide total / num_instances, and
///     num_instances must divide total.
/// Returns std::nullopt when the bound is none of these forms over a
/// ktdf.tiling.reserve_size; such a trip count cannot be constrained.
[[nodiscard]] auto getSplitRequirement(Value upper_bound, int64_t num_instances)
    -> std::optional<SplitRequirement>;

/// The constraints on one ktdf.tiling.reserve_size: its own min_value and
/// divisibility, the totals of the loops it tiles, and the requirements of the
/// ktdf.parallel loops whose trip counts it determines (see
/// getSplitRequirement).
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

  /// Returns true if `tile_size` satisfies every constraint, and also
  /// `extra`.
  [[nodiscard]] auto isLegal(int64_t tile_size,
                             const TileSizeRequirement& extra = {}) const
      -> bool;

  /// Returns every legal tile size (see isLegal), ascending. A legal tile size
  /// divides every non-zero total, so the smallest non-zero total bounds the
  /// set. A loop with no iterations does not constrain the tile size; if no
  /// loop has iterations and nothing else bounds the tile size, every multiple
  /// of the divisibility from min_value up is legal and only the smallest is
  /// returned.
  [[nodiscard]] auto legalTileSizes(const TileSizeRequirement& extra = {}) const
      -> llvm::SmallVector<int64_t>;

  /// Returns true if some tile size is legal.
  [[nodiscard]] auto isFeasible(const TileSizeRequirement& extra = {}) const
      -> bool {
    return !legalTileSizes(extra).empty();
  }

 private:
  TilingReserveSizeOp reserve_size_op_;
  int64_t min_value_;
  int64_t divisibility_;
  llvm::SmallVector<TiledLoop> tiled_loops_;
  /// Requirements of the ktdf.parallel loops built from this tile size.
  TileSizeRequirement split_requirement_;
  bool split_impossible_ = false;
  llvm::SmallVector<llvm::StringRef> unrecognized_uses_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_TILESIZECONSTRAINTS_H_
