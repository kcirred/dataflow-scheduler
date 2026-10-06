// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances -tile-size-selection -canonicalize | FileCheck %s --check-prefix=TILED

// Two dynamic point loops. The outer one's trip count is
// derive_size(%tile_p) / 64, and %tile_p can only be 64 (a multiple of 64 that
// divides 64), so that loop always runs once: no tile size makes it a multiple
// of num_instances = 2, and it is not split. The inner one's trip count is
// %tile_d itself, which can be even, so it is split; the ktdf.parallel then
// requires %tile_d to be a multiple of 2, which tile size selection honors.

// CHECK-LABEL:   func.func @split_skips_single_iteration_loop() {
// CHECK:           %[[BLOCKS:.*]] = arith.divui %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      scf.for %[[I:[a-zA-Z0-9_]+]] = %{{.*}} to %[[BLOCKS]] step %{{.*}} {
// CHECK-NEXT:        ktdf.parallel (%[[J:[a-zA-Z0-9_]+]], %{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK-NEXT:          ktdf.pipeline {
// CHECK:                 "test.body"(%[[I]], %[[J]]) : (index, index) -> ()
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}

// After tile size selection %tile_p is 64 (its loop folds away) and %tile_d is
// 2, so the split loop runs 2 iterations, one per instance.
// TILED-LABEL:   func.func @split_skips_single_iteration_loop() {
// TILED-NOT:       ktdf.tiling.reserve_size
// TILED-DAG:       %[[C2:.*]] = arith.constant 2 : index
// TILED:           %[[N:.*]] = ktdf.tiling.derive_size [%{{.*}} : %[[C2]]], total_size = %{{.*}} : index
// TILED-NEXT:      ktdf.parallel (%{{.*}}, %{{.*}}) = (%{{.*}}) to (%[[N]]) step (%{{.*}}) distribute(num_instances = 2) {

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @split_skips_single_iteration_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %tile_p = ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index} : index
    %tile_d = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %num_p = arith.ceildivui %c64, %tile_p : index
    scf.for %tp = %c0 to %num_p step %c1 {
      %num_d = arith.ceildivui %c128, %tile_d : index
      %size_p = ktdf.tiling.derive_size [%tp : %tile_p], total_size = %c64 : index
      scf.for %td = %c0 to %num_d step %c1 {
        %size_d = ktdf.tiling.derive_size [%td : %tile_d], total_size = %c128 : index
        %blocks_p = arith.divui %size_p, %c64 : index
        scf.for %i = %c0 to %blocks_p step %c1 {
          scf.for %j = %c0 to %size_d step %c1 {
            ktdf.pipeline {
              ktdf.stage depends_in(none) depends_out(none) {
                "test.body"(%i, %j) : (index, index) -> ()
              } {applicable_units = ["SFU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}
