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

#include <cstdint>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// Divisibility requirements for a positive, signed 64-bit tile size.
/// `required_multiple = 4` and `must_divide = 24`, for example, admit tile
/// sizes 4, 8, 12 and 24. A zero must_divide imposes no upper bound: every
/// positive integer divides zero, so zero-iteration loops add no restriction.
struct TileSizeRequirement {
  int64_t required_multiple = 1;
  int64_t must_divide = 0;
  /// True for contradictory requirements, invalid inputs or arithmetic
  /// overflow.
  bool impossible = false;

  /// Returns the intersection of the two sets of permitted tile sizes.
  /// Contradictions and unrepresentable least common multiples are impossible.
  [[nodiscard]] auto combine(const TileSizeRequirement& other) const
      -> TileSizeRequirement;
};

/// What distributing a loop across instances requires of the tile size its
/// trip count is built from.
struct SplitRequirement {
  TilingReserveSizeOp reserve_size_op;
  TileSizeRequirement requirement;
};

/// Returns a sufficient tile-size requirement for distributing a zero-based,
/// unit-step loop equally across `num_instances` hardware instances.
/// `upper_bound` is the loop's exclusive upper bound, in iterations:
///   - ktdf.tiling.derive_size [... : %tile] (a point loop): the trip count is
///     the tile size, so it must be a multiple of num_instances;
///   - arith.divui (that derive_size, F) for a constant F: the trip count is
///     tile / F, so the tile must be a multiple of F * num_instances;
///   - arith.ceildivui (total, %tile) (a tile loop): the trip count is
///     total / tile, so the tile must divide total / num_instances, and
///     num_instances must divide total.
/// Only single-level derive_size operations with a non-negative constant
/// total are supported. Returns std::nullopt for other bound forms; the
/// splitter must leave those loops serial. A recognized but infeasible bound
/// returns a SplitRequirement whose requirement.impossible flag is true.
[[nodiscard]] auto getSplitRequirement(Value upper_bound, int64_t num_instances)
    -> std::optional<SplitRequirement>;

/// A snapshot of the legal values for a reserved tile-size placeholder.
/// A tile groups iterations for processing together. A placeholder may size
/// several loops; its value must respect its minimum and divisibility, divide
/// every tiled total, and permit equal work across distributed instances.
/// Rebuild this analysis after changing the placeholder's attributes or uses.
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
  /// Unsupported ceildivui uses that prevent proving a tile size legal;
  /// one reason per use, for diagnostics.
  [[nodiscard]] auto getUnsupportedUseReasons() const
      -> llvm::ArrayRef<llvm::StringRef> {
    return unsupported_use_reasons_;
  }

  /// Returns true if the tile size tiles at least one loop. Without one there
  /// is no total to divide, and so nothing to choose a tile size against.
  [[nodiscard]] auto hasTiledLoops() const -> bool { return has_tiled_loops_; }

  /// Returns true if `tile_size` satisfies every constraint, and also
  /// `extra`.
  [[nodiscard]] auto isLegal(int64_t tile_size,
                             const TileSizeRequirement& extra = {}) const
      -> bool;

  /// Returns the smallest legal positive tile size, or no value if none exists.
  /// Zero-iteration loops impose no upper bound. Divisor pairs are searched
  /// only when rounding min_value up to a legal multiple is insufficient.
  [[nodiscard]] auto smallestLegalTileSize(
      const TileSizeRequirement& extra = {}) const -> std::optional<int64_t>;

  /// Tests feasibility without enumerating tile sizes: when a total bounds
  /// the size, the greatest common divisor of all totals is the largest
  /// possible tile. Otherwise only the minimum and divisibility constrain it.
  [[nodiscard]] auto isFeasible(const TileSizeRequirement& extra = {}) const
      -> bool;

 private:
  /// Adds constraints from ceildivui bounds used by serial or parallel loops.
  void collectTiledLoopConstraints();
  /// Adds requirements imposed by existing distributed loops in this scope.
  void collectParallelLoopConstraints();

  /// Combines the precomputed constraints with a proposed additional split.
  [[nodiscard]] auto effectiveRequirement(
      const TileSizeRequirement& extra) const -> TileSizeRequirement;

  TilingReserveSizeOp reserve_size_op_;
  int64_t min_value_;
  int64_t divisibility_;
  bool has_tiled_loops_ = false;
  TileSizeRequirement collected_requirement_;
  llvm::SmallVector<llvm::StringRef> unsupported_use_reasons_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_TILESIZECONSTRAINTS_H_
