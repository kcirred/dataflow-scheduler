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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::ktdf;

namespace {

/// Returns the smallest representable multiple at or above a positive minimum.
/// Both arguments must be positive. No result means rounding would overflow.
auto roundUpToMultiple(int64_t minimum, int64_t multiple)
    -> std::optional<int64_t> {
  assert(minimum > 0 && multiple > 0);
  const int64_t remainder = minimum % multiple;
  if (remainder == 0) return minimum;
  const int64_t increment = multiple - remainder;
  if (minimum > std::numeric_limits<int64_t>::max() - increment)
    return std::nullopt;
  return minimum + increment;
}

}  // namespace

auto TileSizeRequirement::combine(const TileSizeRequirement& other) const
    -> TileSizeRequirement {
  if (impossible || other.impossible || required_multiple < 1 ||
      other.required_multiple < 1 || must_divide < 0 || other.must_divide < 0)
    return {1, 0, true};
  const int64_t reduced_multiple =
      required_multiple / std::gcd(required_multiple, other.required_multiple);
  if (reduced_multiple >
      std::numeric_limits<int64_t>::max() / other.required_multiple)
    return {1, 0, true};
  TileSizeRequirement result;
  result.required_multiple = reduced_multiple * other.required_multiple;
  // gcd(0, d) == d: "no bound" combines as the identity.
  result.must_divide = std::gcd(must_divide, other.must_divide);
  result.impossible = result.must_divide != 0 &&
                      result.must_divide % result.required_multiple != 0;
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
    if (!reserve_size_op || !total.has_value() || *total < 0)
      return std::nullopt;
    SplitRequirement result{reserve_size_op, {}};
    if (*total % num_instances != 0) {
      result.requirement.impossible = true;
      return result;
    }
    // total / tile must be a multiple of N, i.e. tile divides total / N.
    result.requirement.must_divide = *total / num_instances;
    return result;
  }

  // A point loop: derive_size, or derive_size / F for a constant F.
  int64_t elements_per_iteration = 1;
  Value point_count = upper_bound;
  if (auto divui_op = upper_bound.getDefiningOp<arith::DivUIOp>()) {
    std::optional<int64_t> divisor = getConstantIntValue(divui_op.getRhs());
    if (!divisor.has_value() || *divisor < 1) return std::nullopt;
    elements_per_iteration = *divisor;
    point_count = divui_op.getLhs();
  }
  auto derive_op = point_count.getDefiningOp<TilingDeriveSizeOp>();
  if (!derive_op || derive_op.getTileSizes().size() != 1) return std::nullopt;
  auto total = getConstantIntValue(derive_op.getTotalSize());
  if (!total || *total < 0) return std::nullopt;
  // With a single tiling level and no remainder, the point count equals
  // the reserved tile size.
  auto reserve_size_op =
      derive_op.getTileSizes().front().getDefiningOp<TilingReserveSizeOp>();
  if (!reserve_size_op) return std::nullopt;
  SplitRequirement result{reserve_size_op, {}};
  if (elements_per_iteration >
      std::numeric_limits<int64_t>::max() / num_instances) {
    result.requirement.impossible = true;
    return result;
  }
  result.requirement.required_multiple = elements_per_iteration * num_instances;
  // The actual point-loop size is a remainder on the last tile unless the
  // selected tile divides this derive_size's own total.
  result.requirement.must_divide = *total;
  return result;
}

TileSizeConstraints::TileSizeConstraints(TilingReserveSizeOp reserve_size_op)
    : reserve_size_op_(reserve_size_op),
      min_value_(reserve_size_op.getMinValue().getSExtValue()),
      divisibility_(reserve_size_op.getDivisibility().getSExtValue()) {
  collected_requirement_.required_multiple = divisibility_;
  collectTiledLoopConstraints();
  collectParallelLoopConstraints();
}

void TileSizeConstraints::collectTiledLoopConstraints() {
  Value tile_size = reserve_size_op_.getResult();
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
      unsupported_use_reasons_.push_back(
          "tile size used as dividend (LHS) in ceildivui, expected divisor "
          "(RHS)");
      continue;
    }
    std::optional<int64_t> total_size =
        getConstantIntValue(ceildiv_op.getLhs());
    if (!total_size.has_value() || *total_size < 0) {
      unsupported_use_reasons_.push_back(
          "expected a non-negative constant total size in arith.ceildivui");
      continue;
    }

    Value bound = ceildiv_op.getResult();
    for (Operation* bound_user : bound.getUsers()) {
      auto for_op = dyn_cast<scf::ForOp>(bound_user);
      auto parallel_op = dyn_cast<ParallelOp>(bound_user);
      const bool is_loop_bound =
          (for_op && for_op.getUpperBound() == bound) ||
          (parallel_op &&
           llvm::is_contained(parallel_op.getUpperBounds(), bound));
      if (!is_loop_bound) continue;
      has_tiled_loops_ = true;
      collected_requirement_ = collected_requirement_.combine({1, *total_size});
    }
  }
}

void TileSizeConstraints::collectParallelLoopConstraints() {
  // Every ktdf.parallel whose trip count this tile size determines must run
  // the same number of iterations on each instance.
  Operation* scope = reserve_size_op_->getParentOp();
  while (scope && !scope->hasTrait<OpTrait::IsIsolatedFromAbove>())
    scope = scope->getParentOp();
  if (!scope) return;
  scope->walk([&](ParallelOp parallel_op) {
    for (Value upper_bound : parallel_op.getUpperBounds()) {
      auto split =
          getSplitRequirement(upper_bound, parallel_op.getNumInstances());
      if (!split || split->reserve_size_op != reserve_size_op_) continue;
      collected_requirement_ =
          collected_requirement_.combine(split->requirement);
    }
  });
}

auto TileSizeConstraints::effectiveRequirement(
    const TileSizeRequirement& extra) const -> TileSizeRequirement {
  TileSizeRequirement effective = collected_requirement_.combine(extra);
  if (!unsupported_use_reasons_.empty()) effective.impossible = true;
  return effective;
}

auto TileSizeConstraints::isLegal(int64_t tile_size,
                                  const TileSizeRequirement& extra) const
    -> bool {
  if (tile_size < std::max<int64_t>(1, min_value_)) return false;
  const auto required = effectiveRequirement(extra);
  return !required.impossible && tile_size % required.required_multiple == 0 &&
         (required.must_divide == 0 || required.must_divide % tile_size == 0);
}

auto TileSizeConstraints::isFeasible(const TileSizeRequirement& extra) const
    -> bool {
  const auto required = effectiveRequirement(extra);
  if (required.impossible) return false;
  if (required.must_divide != 0) return isLegal(required.must_divide, extra);
  // With no nonzero total, any representable multiple above the minimum works.
  const int64_t minimum = std::max<int64_t>(1, min_value_);
  return roundUpToMultiple(minimum, required.required_multiple).has_value();
}

auto TileSizeConstraints::smallestLegalTileSize(
    const TileSizeRequirement& extra) const -> std::optional<int64_t> {
  if (!isFeasible(extra)) return std::nullopt;
  const auto required = effectiveRequirement(extra);
  const auto first_multiple = roundUpToMultiple(
      std::max<int64_t>(1, min_value_), required.required_multiple);
  assert(first_multiple &&
         "feasible constraints must have a representable multiple");
  if (isLegal(*first_multiple, extra)) return first_multiple;

  // Feasibility guarantees a nonzero total here, and that total is legal.
  // A legal size is required_multiple * factor, where factor divides
  // must_divide / required_multiple. Search these dimensionless factors,
  // rather than every element count. The square-root bound is inclusive;
  // division in the condition avoids overflowing factor * factor.
  int64_t smallest = required.must_divide;
  const int64_t factor_total =
      required.must_divide / required.required_multiple;
  for (int64_t factor = 1; factor <= factor_total / factor; ++factor) {
    if (factor_total % factor != 0) continue;
    for (int64_t candidate_factor : {factor, factor_total / factor}) {
      // Both factors divide factor_total, so the product cannot exceed
      // must_divide.
      const int64_t candidate = candidate_factor * required.required_multiple;
      if (candidate >= *first_multiple && candidate < smallest)
        smallest = candidate;
    }
  }
  return smallest;
}
