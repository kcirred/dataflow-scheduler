// RUN: dataflow-scheduler-opt --tile-size-selection --tile-sizes=2 --allow-unregistered-dialect %s | FileCheck %s
// RUN: not dataflow-scheduler-opt --tile-size-selection --tile-sizes=3 --allow-unregistered-dialect %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not dataflow-scheduler-opt --tile-size-selection --tile-sizes=0 --allow-unregistered-dialect %s 2>&1 | FileCheck %s --check-prefix=ZERO

// Six elements admit tiles of size two and three, but distributing each tile's
// elements over two instances rules out three. An override must honor the split.
// CHECK-LABEL: func.func @override
// CHECK: %[[TILE:.*]] = arith.constant 2 : index
// CHECK: ktdf.tiling.derive_size [%{{.*}} : %[[TILE]]]
// INVALID: requested tile size 3 violates the reserved tile or parallel-loop constraints
// ZERO: requested tile size 0 violates the reserved tile or parallel-loop constraints
func.func @override() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %total = arith.constant 6 : index
  %tile = ktdf.tiling.reserve_size {min_value = 1 : index, divisibility = 1 : index} : index
  %bound = arith.ceildivui %total, %tile : index
  scf.for %t = %c0 to %bound step %c1 {
    %size = ktdf.tiling.derive_size [%t : %tile], total_size = %total : index
    ktdf.parallel (%i, %instance) = (%c0) to (%size) step (%c1) distribute(num_instances = 2) {
      "test.body"(%i) : (index) -> ()
      ktdf.parallel_yield
    }
  }
  return
}
