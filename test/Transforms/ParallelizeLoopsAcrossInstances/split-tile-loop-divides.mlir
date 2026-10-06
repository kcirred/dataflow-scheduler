// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances -tile-size-selection -canonicalize | FileCheck %s --check-prefix=TILED

// A tile loop split across num_instances = 2: its trip count is 6 / %tile, so
// %tile must divide 6 / 2 = 3. On its own tile size selection would pick 2
// (6 / 2 = 3 iterations, odd); the ktdf.parallel's requirement makes it pick 1
// instead, so the loop runs 6 iterations, 3 per instance.

// CHECK-LABEL:   func.func @split_tile_loop_divides() {
// CHECK:           %[[TILE:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:      %[[BOUND:.*]] = arith.ceildivui %{{.*}}, %[[TILE]] : index
// CHECK-NEXT:      ktdf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}) to (%[[BOUND]]) step (%{{.*}}) distribute(num_instances = 2) {

// TILED-LABEL:   func.func @split_tile_loop_divides() {
// TILED-DAG:       %[[C6:.*]] = arith.constant 6 : index
// TILED:           ktdf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}) to (%[[C6]]) step (%{{.*}}) distribute(num_instances = 2) {

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @split_tile_loop_divides() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %tile = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %bound = arith.ceildivui %c6, %tile : index
    scf.for %t = %c0 to %bound step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%t) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
