// RUN: dataflow-scheduler-opt -allow-unregistered-dialect  -parallelize-loops-across-instances %s | FileCheck %s

// Loop bounds depend on a function argument. The trip count is not built from
// a tile size, so nothing can make it a multiple of num_instances, and
// lowering also requires a constant trip count to distribute. The loop is not
// split.

// CHECK-LABEL:   func.func @dynamic_trip(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           "test.body"(%[[VAL_0]]) : (index) -> ()
// CHECK-NEXT:         } {applicable_units = ["SFU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NOT:      ktdf.parallel

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @dynamic_trip(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %arg0 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}