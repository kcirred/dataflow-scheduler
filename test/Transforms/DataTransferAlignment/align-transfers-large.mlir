// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// CHECK-LABEL: func.func @local_schedule_0()

// The strided input view is read through the transposition of its contiguous
// dim (d1, stride 1) and its lane (d2): their strides are exchanged.
// CHECK:      ktdp.construct_memory_view %{{.*}}, sizes: [12, 64, 64], strides: [4096, 64, 1]

// The tile of the aligned dimension is constrained to whole 64-element blocks,
// so tile size selection cannot split one:
// CHECK:      %[[TS0:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index}
// CHECK:      %[[TS1:.*]] = ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index}

// Staging allocs reshaped to memref<?x?x1x64x64xf16, "L1">, one 64x64 block per
// 64 points of the aligned tile:
// CHECK:      %[[PRIV:.*]]:4 = ktdf.private -> (memref<?x?x1x64x64xf16, "L1">, memref<?x?x1x64x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK:        %[[NB0:.*]] = arith.divui %[[TS1]], %{{.*}} : index
// CHECK-NEXT:   %[[A0:.*]] = memref.alloc(%[[TS0]], %[[NB0]]) : memref<?x?x1x64x64xf16, "L1">
// CHECK:        %[[NB1:.*]] = arith.divui %[[TS1]], %{{.*}} : index
// CHECK-NEXT:   %[[A1:.*]] = memref.alloc(%[[TS0]], %[[NB1]]) : memref<?x?x1x64x64xf16, "L1">
// CHECK:        ktdf.private_yield %[[A0]], %[[A1]]

// MNILU Stage: per-tile trip count divided by 64, each iteration advances one
// 64-point block, transfer widened to [1, 64, 64]. On the swapped view the
// block index of d1 moves to d2 with its stride.
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          %[[NB:.*]] = arith.divui %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     scf.for %[[ARG3:.*]] = %{{.*}} to %[[NB]] step %{{.*}} {
// CHECK:            %[[IDX:.*]] = ktdf.tiling.linearize_index [%{{.*}} : %[[TS1]]], [%[[ARG3]] : %c64{{.*}}] : index
// CHECK-NEXT:       ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}, %[[IDX]]] size [1, 64, 64] to %[[PRIV]]#0[%[[ARG2]], %[[ARG3]], 0, 0, 0] size [1, 1, 1, 64, 64]

// Middle Stage: nested pipeline wrapped with element loops for scalar
// processing; the element transfers keep the block indices of the stage loops
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          scf.for %[[ARG3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:              scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:                ktdf.pipeline {
// CHECK:                  ktdf.stage {{.*}} {
// CHECK:                    ktdf.data_transfer from %[[PRIV]]#0[%[[ARG2]], %[[ARG3]], 0, %[[ARG4]], %[[ARG5]]] size [1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:                  }
// CHECK:                  ktdf.stage
// CHECK:                  ktdf.stage {{.*}} {
// CHECK:                    ktdf.data_transfer from %{{.*}} size [64] to %[[PRIV]]#1[%[[ARG2]], %[[ARG3]], 0, %[[ARG5]], %[[ARG4]]] size [1, 1, 1, 1, 1] {transfer_mode = "lane0"}
// CHECK:                  }
// CHECK:                }
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }

// MNISU Stage: per-tile trip count divided by 64, each iteration advances one
// 64-point block, transfer widened to [1, 64, 64]
// CHECK:      ktdf.stage
// CHECK:        scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          %[[NB:.*]] = arith.divui %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     scf.for %[[ARG3:.*]] = %{{.*}} to %[[NB]] step %{{.*}} {
// CHECK:            %[[IDX:.*]] = ktdf.tiling.linearize_index [%{{.*}} : %[[TS1]]], [%[[ARG3]] : %c64{{.*}}] : index
// CHECK-NEXT:       ktdf.data_transfer from %[[PRIV]]#1[%[[ARG2]], %[[ARG3]], 0, 0, 0] size [1, 1, 1, 64, 64] to %{{.*}}[%{{.*}}, %[[IDX]], %{{.*}}] size [1, 64, 64]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @identity_0() attributes {grid = [6]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [6]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c162304 = arith.constant 162304 : index
      %c64000 = arith.constant 64000 : index
      %c12 = arith.constant 12 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [12, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16, strided<[4096, 1, 64]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<12x64x64xf16, strided<[4096, 1, 64]>> to memref<12x64x64xf16, strided<[4096, 1, 64]>, "DDR">
      %cast = memref.cast %memspacecast : memref<12x64x64xf16, strided<[4096, 1, 64]>, "DDR"> to memref<12x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<12x64x64xf16> to memref<12x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [12, 64, 64], strides: [4096, 64, 1] : memref<12x64x64xf16, "DDR"> to memref<12x64x64xf16, strided<[4096, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<12x64x64xf16, strided<[4096, 64, 1]>, "DDR"> to memref<12x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = arith.ceildivui %c12, %2 : index
      scf.for %arg0 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c64, %3 : index
        %6 = ktdf.tiling.derive_size [%arg0 : %2], total_size = %c12 : index
        scf.for %arg1 = %c0 to %5 step %c1 {
          %7 = ktdf.tiling.derive_size [%arg1 : %3], total_size = %c64 : index
          ktdf.pipeline {
            %8:4 = ktdf.private -> (memref<?x?x1x1x64xf16, "L1">, memref<?x?x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
              %alloc = memref.alloc(%2, %3) : memref<?x?x1x1x64xf16, "L1">
              %alloc_2 = memref.alloc(%2, %3) : memref<?x?x1x1x64xf16, "L1">
              %9 = ktdf.create_token : !ktdf.token
              %10 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %alloc, %alloc_2, %9, %10 : memref<?x?x1x1x64xf16, "L1">, memref<?x?x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%8#2) {
              scf.for %arg2 = %c0 to %6 step %c1 {
                %9 = ktdf.tiling.linearize_index [%arg0 : %2], [%arg2 : %c1] : index
                scf.for %arg3 = %c0 to %7 step %c1 {
                  %10 = ktdf.tiling.linearize_index [%arg1 : %3], [%arg3 : %c1] : index
                  ktdf.data_transfer from %cast[%9, %10, %c0 * 64] size [1, 1, 64] to %8#0[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">, memref<?x?x1x1x64xf16, "L1">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["MNILU"]}
            ktdf.stage depends_in(%8#2) depends_out(%8#3) {
              scf.for %arg2 = %c0 to %6 step %c1 {
                scf.for %arg3 = %c0 to %7 step %c1 {
                  ktdf.pipeline {
                    %9:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                      %10 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      %11 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                      %12 = ktdf.create_token : !ktdf.token
                      %13 = ktdf.create_token : !ktdf.token
                      ktdf.private_yield %10, %11, %12, %13 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                    }
                    ktdf.stage depends_in(none) depends_out(%9#2) {
                      ktdf.data_transfer from %8#0[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] to %9#0 size [64] : memref<?x?x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    } {applicable_units = ["L1LU"]}
                    ktdf.stage depends_in(%9#2) depends_out(%9#3) {
                      %10 = ktdf.read_from_fifo %9#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                      ktdf.write_to_fifo %10, %9#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    } {applicable_units = ["SFU"]}
                    ktdf.stage depends_in(%9#3) depends_out(none) {
                      ktdf.data_transfer from %9#1 size [64] to %8#1[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x64xf16, "L1">
                    } {applicable_units = ["L1SU"]}
                  }
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["L1LU", "SFU", "L1SU"]}
            ktdf.stage depends_in(%8#3) depends_out(none) {
              scf.for %arg2 = %c0 to %6 step %c1 {
                %9 = ktdf.tiling.linearize_index [%arg0 : %2], [%arg2 : %c1] : index
                scf.for %arg3 = %c0 to %7 step %c1 {
                  %10 = ktdf.tiling.linearize_index [%arg1 : %3], [%arg3 : %c1] : index
                  ktdf.data_transfer from %8#1[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] to %cast_1[%9, %10, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x64xf16, "L1">, memref<12x64x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["MNISU"]}
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
