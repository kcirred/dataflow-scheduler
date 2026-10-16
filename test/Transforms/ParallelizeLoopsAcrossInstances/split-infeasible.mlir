// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// The only parallel loop's trip count is %tile, which must be a multiple of 3
// and divide 3, so it is always 3: no tile size makes it a multiple of
// num_instances = 2. The loop is not split.

// CHECK-LABEL:   func.func @split_infeasible() {
// CHECK-NOT:       ktdf.parallel
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             "test.body"
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NOT:       ktdf.parallel

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @split_infeasible() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %tile = ktdf.tiling.reserve_size {divisibility = 3 : index, min_value = 3 : index} : index
    %num = arith.ceildivui %c3, %tile : index
    scf.for %t = %c0 to %num step %c1 {
      %size = ktdf.tiling.derive_size [%t : %tile], total_size = %c3 : index
      scf.for %i = %c0 to %size step %c1 {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            "test.body"(%i) : (index) -> ()
          } {applicable_units = ["SFU"]}
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }
}
