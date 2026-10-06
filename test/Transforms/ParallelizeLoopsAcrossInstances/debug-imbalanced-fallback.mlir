// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// Verify the debug log says the imbalanced loop (trip count 5, num_instances
// 2) is not split: lowering distributes only a trip count that is a multiple
// of num_instances.

// DEBUG-NOT: imbalanced (fallback)
// DEBUG: [parallelize-loops-across-instances {{.*}}] skip pipeline at loc({{.*}}): no parallel loop in scope satisfies requirements

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @debug_imbalanced() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.for %i = %c0 to %c5 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i) : (index) -> ()
        } {applicable_units = ["SFU"]}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}