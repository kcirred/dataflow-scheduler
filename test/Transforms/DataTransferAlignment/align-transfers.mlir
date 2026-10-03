// RUN: dataflow-scheduler-opt --data-transfer-alignment --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @local_schedule_0()

// Staging allocs reshaped to 1x1x64x64:
// CHECK:      %[[PRIV:.*]]:4 = ktdf.private -> (memref<1x1x64x64xf16, "L1">, memref<1x1x64x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-DAG:    %[[A0:.*]] = memref.alloc() : memref<1x1x64x64xf16, "L1">
// CHECK-DAG:    %[[A1:.*]] = memref.alloc() : memref<1x1x64x64xf16, "L1">
// CHECK:        ktdf.private_yield %[[A0]], %[[A1]]

// MNILU Stage: loop bound adjusted to 1, transfer widened to [1, 64, 64]
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          ktdf.data_transfer from %{{.*}} size [1, 64, 64] to %[[PRIV]]#0[%[[ARG0]], 0, 0, 0] size [1, 1, 64, 64]

// Middle Stage: nested pipeline wrapped with element loops for scalar processing
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          scf.for %[[ARG1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:              ktdf.pipeline {
// CHECK:                ktdf.stage {{.*}} {
// CHECK:                  ktdf.data_transfer from %[[PRIV]]#0[0, 0, %[[ARG1]], %[[ARG2]]] size [1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:                }
// CHECK:                ktdf.stage
// CHECK:                ktdf.stage {{.*}} {
// CHECK:                  ktdf.data_transfer from %{{.*}} size [64] to %[[PRIV]]#1[0, 0, %[[ARG2]], %[[ARG1]]] size [1, 1, 1, 1] {transfer_mode = "lane0"}
// CHECK:                }
// CHECK:              }

// MNISU Stage: loop bound adjusted to 1, transfer widened to [1, 64, 64]
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          ktdf.data_transfer from %[[PRIV]]#1[%[[ARG0]], 0, 0, 0] size [1, 1, 64, 64] to %{{.*}} size [1, 64, 64]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @identity_0() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c162304 = arith.constant 162304 : index
      %c64000 = arith.constant 64000 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [1, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16, strided<[4096, 1, 64]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<1x64x64xf16, strided<[4096, 1, 64]>> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR">
      %cast = memref.cast %memspacecast : memref<1x64x64xf16, strided<[4096, 1, 64]>, "DDR"> to memref<1x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64x64xf16> to memref<1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64, 64], strides: [4096, 64, 1] : memref<1x64x64xf16, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %alloc_2 = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_2, %3, %4 : memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%c0, %arg0, %c0 * 64] size [1, 1, 64] to %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">, memref<64x1x1x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<64x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                ktdf.write_to_fifo %4, %3#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x1x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %cast_1[%c0, %arg0, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x1x64xf16, "L1">, memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}

// -----

// The same pipeline with a row-major input view: every transfer is already
// stick-aligned, so the pass leaves the staging allocs, loops and transfers as
// they are, and sets no transfer_mode.

// CHECK-LABEL: func.func @local_schedule_0()

// CHECK:      %[[PRIV:.*]]:4 = ktdf.private -> (memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {

// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:     ktdf.data_transfer from %{{.*}}[%{{.*}}, %[[ARG0]], %{{.*}} * 64] size [1, 1, 64] to %[[PRIV]]#0[%[[ARG0]], 0, 0, 0] size [1, 1, 1, 64]

// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK:            ktdf.data_transfer from %[[PRIV]]#0[%[[ARG0]], 0, 0, 0] size [1, 1, 1, 64] to %{{.*}} size [64] :
// CHECK:            ktdf.data_transfer from %{{.*}} size [64] to %[[PRIV]]#1[%[[ARG0]], 0, 0, 0] size [1, 1, 1, 64] :

// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:     ktdf.data_transfer from %[[PRIV]]#1[%[[ARG0]], 0, 0, 0] size [1, 1, 1, 64] to %{{.*}}[%{{.*}}, %[[ARG0]], %{{.*}} * 64] size [1, 1, 64]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @identity_0() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c162304 = arith.constant 162304 : index
      %c64000 = arith.constant 64000 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
      %1 = ktdp.construct_memory_view %c162304, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<1x64x64xf16> to memref<1x64x64xf16, "DDR">
      %cast = memref.cast %memspacecast : memref<1x64x64xf16, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64x64xf16> to memref<1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64, 64], strides: [4096, 64, 1] : memref<1x64x64xf16, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<1x64x64xf16, strided<[4096, 64, 1]>, "DDR"> to memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %alloc_2 = memref.alloc() : memref<64x1x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_2, %3, %4 : memref<64x1x1x64xf16, "L1">, memref<64x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%c0, %arg0, %c0 * 64] size [1, 1, 64] to %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">, memref<64x1x1x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<64x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                ktdf.write_to_fifo %4, %3#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64x1x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0, 0, 0] size [1, 1, 1, 64] to %cast_1[%c0, %arg0, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x1x64xf16, "L1">, memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
