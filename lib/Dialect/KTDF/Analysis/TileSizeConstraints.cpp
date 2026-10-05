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

TileSizeConstraints::TileSizeConstraints(TilingReserveSizeOp reserve_size_op)
    : reserve_size_op_(reserve_size_op),
      min_value_(reserve_size_op.getMinValue().getSExtValue()),
      divisibility_(std::max<int64_t>(
          1, reserve_size_op.getDivisibility().getSExtValue())) {
  Value tile_size = reserve_size_op.getResult();
  for (Operation* user : tile_size.getUsers()) {
    // Pattern: %bound = arith.ceildivui %total_size, %tile_size
    // Then: scf.for %i = %c0 to %bound step %c1
    auto ceildiv_op = dyn_cast<arith::CeilDivUIOp>(user);
    // Other uses (derive_size, linearize_index, allocations) follow from the
    // tile size and do not constrain it.
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
    }
  }
}

auto TileSizeConstraints::isLegal(int64_t tile_size,
                                  int64_t extra_divisibility) const -> bool {
  if (tile_size < std::max<int64_t>(1, min_value_)) return false;
  int64_t divisibility =
      std::lcm(divisibility_, std::max<int64_t>(1, extra_divisibility));
  if (tile_size % divisibility != 0) return false;
  return llvm::all_of(tiled_loops_, [&](const TiledLoop& tiled) {
    return tiled.total_size % tile_size == 0;
  });
}

auto TileSizeConstraints::legalTileSizes(int64_t extra_divisibility) const
    -> llvm::SmallVector<int64_t> {
  int64_t divisibility =
      std::lcm(divisibility_, std::max<int64_t>(1, extra_divisibility));
  // The smallest multiple of the divisibility from min_value up.
  int64_t lowest = std::max<int64_t>(1, min_value_);
  lowest = (lowest + divisibility - 1) / divisibility * divisibility;

  int64_t highest = std::numeric_limits<int64_t>::max();
  for (const TiledLoop& tiled : tiled_loops_)
    if (tiled.total_size > 0) highest = std::min(highest, tiled.total_size);
  if (highest == std::numeric_limits<int64_t>::max()) return {lowest};

  llvm::SmallVector<int64_t> legal;
  for (int64_t tile_size = lowest; tile_size <= highest;
       tile_size += divisibility)
    if (isLegal(tile_size, extra_divisibility)) legal.push_back(tile_size);
  return legal;
}
