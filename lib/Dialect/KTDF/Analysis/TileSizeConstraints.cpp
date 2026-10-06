//===-- TileSizeConstraints.cpp ---------------------------------*- c++ -*-===//
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
// This file implements the constraints on a ktdf.tiling.reserve_size.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/TileSizeConstraints.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::ktdf;

auto TileSizeRequirement::combine(const TileSizeRequirement& other) const
    -> TileSizeRequirement {
  TileSizeRequirement result;
  result.multiple_of = std::lcm(std::max<int64_t>(1, multiple_of),
                                std::max<int64_t>(1, other.multiple_of));
  // gcd(0, d) == d: "no bound" combines as the identity.
  result.divides = std::gcd(divides, other.divides);
  return result;
}

auto mlir::ktdf::getSplitRequirement(Value upper_bound, int64_t num_instances)
    -> std::optional<SplitRequirement> {
  if (num_instances < 1) return std::nullopt;

  // A tile loop: arith.ceildivui %total, %tile.
  if (auto ceildiv_op = upper_bound.getDefiningOp<arith::CeilDivUIOp>()) {
    auto reserve_size_op =
        ceildiv_op.getRhs().getDefiningOp<TilingReserveSizeOp>();
    std::optional<int64_t> total = getConstantIntValue(ceildiv_op.getLhs());
    if (!reserve_size_op || !total.has_value()) return std::nullopt;
    SplitRequirement result{reserve_size_op, {}};
    if (*total % num_instances != 0) {
      result.impossible = true;
      return result;
    }
    // total / tile must be a multiple of N, i.e. tile divides total / N.
    result.requirement.divides = *total / num_instances;
    return result;
  }

  // A point loop: derive_size, or derive_size / F for a constant F.
  int64_t factor = 1;
  Value derived = upper_bound;
  if (auto divui_op = upper_bound.getDefiningOp<arith::DivUIOp>()) {
    std::optional<int64_t> divisor = getConstantIntValue(divui_op.getRhs());
    if (!divisor.has_value() || *divisor < 1) return std::nullopt;
    factor = *divisor;
    derived = divui_op.getLhs();
  }
  auto derive_op = derived.getDefiningOp<TilingDeriveSizeOp>();
  if (!derive_op || derive_op.getTileSizes().empty()) return std::nullopt;
  // The point loop's trip count is its innermost tile size (the epilogue
  // remainder is excluded by the tile dividing the total).
  auto reserve_size_op =
      derive_op.getTileSizes().back().getDefiningOp<TilingReserveSizeOp>();
  if (!reserve_size_op) return std::nullopt;
  SplitRequirement result{reserve_size_op, {}};
  result.requirement.multiple_of = factor * num_instances;
  return result;
}

TileSizeConstraints::TileSizeConstraints(TilingReserveSizeOp reserve_size_op)
    : reserve_size_op_(reserve_size_op),
      min_value_(reserve_size_op.getMinValue().getSExtValue()),
      divisibility_(std::max<int64_t>(
          1, reserve_size_op.getDivisibility().getSExtValue())) {
  Value tile_size = reserve_size_op.getResult();
  for (Operation* user : tile_size.getUsers()) {
    // Pattern: %bound = arith.ceildivui %total_size, %tile_size
    // Then: scf.for %i = %c0 to %bound step %c1, or that loop distributed as
    // a ktdf.parallel.
    auto ceildiv_op = dyn_cast<arith::CeilDivUIOp>(user);
    // Other uses (derive_size, linearize_index, allocations) follow from the
    // tile size and do not constrain it; derive_size bounds of ktdf.parallel
    // loops are collected below.
    // TODO: in future consider memref.alloc sizes to constrain the tile size
    // selection
    if (!ceildiv_op) continue;

    if (ceildiv_op.getRhs() != tile_size) {
      unrecognized_uses_.push_back(
          "tile size used as dividend (LHS) in ceildivui, expected divisor "
          "(RHS)");
      continue;
    }
    std::optional<int64_t> total_size =
        getConstantIntValue(ceildiv_op.getLhs());
    if (!total_size.has_value()) {
      unrecognized_uses_.push_back(
          "non-constant total size in arith.ceildivui");
      continue;
    }

    Value bound = ceildiv_op.getResult();
    for (Operation* bound_user : bound.getUsers()) {
      auto for_op = dyn_cast<scf::ForOp>(bound_user);
      if (for_op && for_op.getUpperBound() == bound)
        tiled_loops_.push_back(TiledLoop{for_op, *total_size});
      auto parallel_op = dyn_cast<ParallelOp>(bound_user);
      if (parallel_op &&
          llvm::is_contained(parallel_op.getUpperBounds(), bound))
        tiled_loops_.push_back(TiledLoop{parallel_op, *total_size});
    }
  }

  // Every ktdf.parallel whose trip count this tile size determines must run
  // the same number of iterations on each instance.
  Operation* scope = reserve_size_op->getParentOp();
  while (scope && !scope->hasTrait<OpTrait::IsIsolatedFromAbove>())
    scope = scope->getParentOp();
  if (!scope) return;
  scope->walk([&](ParallelOp parallel_op) {
    for (Value upper_bound : parallel_op.getUpperBounds()) {
      auto split =
          getSplitRequirement(upper_bound, parallel_op.getNumInstances());
      if (!split || split->reserve_size_op != reserve_size_op) continue;
      if (split->impossible) split_impossible_ = true;
      split_requirement_ = split_requirement_.combine(split->requirement);
    }
  });
}

auto TileSizeConstraints::isLegal(int64_t tile_size,
                                  const TileSizeRequirement& extra) const
    -> bool {
  if (split_impossible_) return false;
  if (tile_size < std::max<int64_t>(1, min_value_)) return false;
  TileSizeRequirement required =
      split_requirement_.combine(extra).combine({divisibility_, 0});
  if (tile_size % required.multiple_of != 0) return false;
  if (required.divides != 0 && required.divides % tile_size != 0) return false;
  return llvm::all_of(tiled_loops_, [&](const TiledLoop& tiled) {
    return tiled.total_size % tile_size == 0;
  });
}

auto TileSizeConstraints::legalTileSizes(const TileSizeRequirement& extra) const
    -> llvm::SmallVector<int64_t> {
  if (split_impossible_) return {};
  TileSizeRequirement required =
      split_requirement_.combine(extra).combine({divisibility_, 0});
  int64_t divisibility = required.multiple_of;
  // The smallest multiple of the divisibility from min_value up.
  int64_t lowest = std::max<int64_t>(1, min_value_);
  lowest = (lowest + divisibility - 1) / divisibility * divisibility;

  int64_t highest = std::numeric_limits<int64_t>::max();
  for (const TiledLoop& tiled : tiled_loops_)
    if (tiled.total_size > 0) highest = std::min(highest, tiled.total_size);
  if (required.divides > 0) highest = std::min(highest, required.divides);
  if (highest == std::numeric_limits<int64_t>::max()) return {lowest};

  llvm::SmallVector<int64_t> legal;
  for (int64_t tile_size = lowest; tile_size <= highest;
       tile_size += divisibility)
    if (isLegal(tile_size, extra)) legal.push_back(tile_size);
  return legal;
}
