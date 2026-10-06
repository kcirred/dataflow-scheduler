//===-- TileSizeSelection.cpp -----------------------------------*- c++ -*-===//
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
// Pass: -tile-size-selection
//
// Resolve ktdf.tiling.reserve_size placeholders to concrete index typed SSA
// values.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/TileSizeConstraints.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "tile-size-selection"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Tile Size Selection pass"),
    llvm::cl::init(false));

static llvm::cl::list<int64_t> ClTileSizes(
    "tile-sizes",
    llvm::cl::desc("Tile sizes for each reserved dimension (global override)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_TILESIZESELECTIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// TODO: should use much large size when proper L1 usage analysis is available.
constexpr int64_t kMaxCandidateTileSize = 2;

void logUnresolved(ktdf::TilingReserveSizeOp reserve_size_op,
                   llvm::StringRef reason) {
  LDBG(1) << "unresolved tiling.reserve_size at " << reserve_size_op.getLoc()
          << ": " << reason;
}

/// Chooses a tile size among those its constraints allow: the largest legal
/// one up to max(kMaxCandidateTileSize, min_value), else the smallest legal one
/// above that. Returns std::nullopt if the tile size tiles no loop (nothing to
/// choose against) or if no tile size is legal.
std::optional<int64_t> chooseTileSize(
    const ktdf::TileSizeConstraints& constraints) {
  ktdf::TilingReserveSizeOp reserve_size_op = constraints.getReserveSizeOp();
  for (llvm::StringRef reason : constraints.getUnrecognizedUses())
    logUnresolved(reserve_size_op, reason);

  if (!constraints.hasTiledLoops()) {
    logUnresolved(reserve_size_op,
                  "no associated loops found via ceildivui pattern");
    return std::nullopt;
  }

  const int64_t min_value = std::max<int64_t>(1, constraints.getMinValue());
  const int64_t start_candidate =
      std::max<int64_t>(kMaxCandidateTileSize, min_value);
  for (int64_t candidate = start_candidate; candidate >= min_value;
       --candidate) {
    LLVM_DEBUG({
      if (candidate <= 1) {
        llvm::dbgs() << "[" PASS_NAME
                     << "] no suitable tile size greater than one for "
                        "tiling.reserve_size at "
                     << reserve_size_op.getLoc() << ".\n";
      }
    });
    if (constraints.isLegal(candidate)) return candidate;
  }
  // The constraints exclude every candidate up to start_candidate (e.g. a
  // divisibility above it); take the smallest legal one above.
  for (int64_t candidate : constraints.legalTileSizes())
    if (candidate > start_candidate) return candidate;
  return std::nullopt;
}

struct TileSizeSelectionPass
    : public ktdf::impl::TileSizeSelectionPassBase<TileSizeSelectionPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    SmallVector<ktdf::TilingReserveSizeOp> reserve_size_ops;
    module.walk([&](ktdf::TilingReserveSizeOp reserve_size_op) {
      reserve_size_ops.push_back(reserve_size_op);
    });

    if (reserve_size_ops.empty()) return;

    ArrayRef<int64_t> effective_tile_sizes(ClTileSizes);

    if (!effective_tile_sizes.empty() &&
        effective_tile_sizes.size() != reserve_size_ops.size()) {
      module.emitError() << "[" PASS_NAME
                         << "] Number of tile sizes specified ("
                         << effective_tile_sizes.size()
                         << ") does not match the number of reserve_size ops ("
                         << reserve_size_ops.size() << ")";
      return signalPassFailure();
    }

    SmallVector<ktdf::TileSizeConstraints> analyses;
    analyses.reserve(reserve_size_ops.size());
    for (ktdf::TilingReserveSizeOp reserve_size_op : reserve_size_ops)
      analyses.emplace_back(reserve_size_op);

    OpBuilder builder(module.getContext());
    for (auto [idx, constraints] : llvm::enumerate(analyses)) {
      ktdf::TilingReserveSizeOp reserve_size_op =
          constraints.getReserveSizeOp();
      std::optional<int64_t> chosen_tile_size;
      if (!effective_tile_sizes.empty()) {
        chosen_tile_size = effective_tile_sizes[idx];
      } else {
        chosen_tile_size = chooseTileSize(constraints);
        if (!chosen_tile_size.has_value() && constraints.hasTiledLoops()) {
          reserve_size_op.emitError()
              << "[" PASS_NAME "] no tile size satisfies min_value = "
              << constraints.getMinValue()
              << ", divisibility = " << constraints.getDivisibility()
              << ", the totals of the loops it tiles and the instances the "
                 "loops it bounds are distributed across";
          return signalPassFailure();
        }
      }
      if (!chosen_tile_size.has_value()) {
        continue;
      }

      builder.setInsertionPoint(reserve_size_op);
      auto constant_op = arith::ConstantIndexOp::create(
          builder, reserve_size_op.getLoc(), *chosen_tile_size);
      reserve_size_op.getResult().replaceAllUsesWith(constant_op.getResult());
      reserve_size_op.erase();
    }

    // TODO: Replace this placeholder heuristic with a real cost model that
    // accounts for L1 memory usage and performance. In particular, a future
    // liveness analysis should determine which L1 buffers are live at each
    // point so we can compute peak live scratchpad usage and choose tile sizes
    // that fully utilize L1 capacity without exceeding the hardware limit.
  }
};

}  // namespace

auto mlir::ktdf::createTileSizeSelectionPass() -> std::unique_ptr<Pass> {
  return std::make_unique<TileSizeSelectionPass>();
}
