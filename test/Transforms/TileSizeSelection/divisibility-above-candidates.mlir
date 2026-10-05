// RUN: dataflow-scheduler-opt --tile-size-selection --allow-unregistered-dialect --canonicalize %s | FileCheck %s

// The tile size below must be a multiple of 4 (divisibility = 4) and divide
// the 64 iterations it tiles, so the legal tile sizes are 4, 8, 16, 32 and 64.
//
// Tile size selection prefers small tiles: it looks for the largest legal size
// up to max(kMaxCandidateTileSize, min_value) = max(2, 1) = 2. Neither 2 nor 1
// is a multiple of 4, so nothing in that range is legal. It then takes the
// smallest legal size above the range, 4, and the tile loop runs 64 / 4 = 16
// times.
//
// Searching only up to 2 used to leave no candidate at all, and the pass
// aborted. A divisibility above that range arises whenever a pass requires a
// tile to be a multiple of something larger, e.g. a loop split across 4
// instances.

// CHECK-LABEL:   func.func private @divisibility_above_candidates() {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C16:.*]] = arith.constant 16 : index
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-NOT:     ktdf.tiling.reserve_size

module {
  func.func private @divisibility_above_candidates() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %tile_size = ktdf.tiling.reserve_size {divisibility = 4 : index, min_value = 1 : index} : index
    %loop_bound = arith.ceildivui %c64, %tile_size : index

    scf.for %i = %c0 to %loop_bound step %c1 {
      %actual_size = ktdf.tiling.derive_size [%i : %tile_size], total_size = %c64 : index
      scf.for %j = %c0 to %actual_size step %c1 {
        "unregistered.op"(%j) : (index) -> ()
      }
    }
    return
  }
}
