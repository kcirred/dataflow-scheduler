// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// A strided view on which every transfer is already legal but whose dims are
// not in memory order: the 3D [128, 64, 128] transpose of dims 0,1, which
// keeps the contiguous dim innermost and only exchanges outer dims. The input
// view's strides [64, 8192, 16384, 1] do not decrease.
//
// Nothing is widened. The view alone is reordered into memory order, keeping
// each dim's size and stride, and the transfers on it are reordered with it.
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK:      ktdp.construct_memory_view %{{.*}}, sizes: [64, 2, 128, 64], strides: [16384, 8192, 64, 1]
// The output view is row-major already and is left alone.
// CHECK:      ktdp.construct_memory_view %{{.*}}, sizes: [128, 2, 64, 64], strides: [8192, 4096, 64, 1]
// No tile is constrained.
// CHECK-NOT:  divisibility = 64
// Load: the view's outer indices are reordered (a, b, c) -> (c, b, a); still
// one stick per transfer.
// CHECK:      ktdf.data_transfer from %{{.*}}[%[[A:[0-9]+]], %[[B:[0-9]+]], %[[C:[0-9]+]], %{{.*}}] size [1, 1, 1, 64] to %{{.*}}#0[%{{.*}}] size [1, 1, 1, 1, 1, 64]
// Store: the row-major output view is unchanged.
// CHECK:      ktdf.data_transfer from %{{.*}}#1[%{{.*}}] size [1, 1, 1, 1, 1, 64] to %{{.*}}[%[[C]], %[[B]], %[[A]], %{{.*}}] size [1, 1, 1, 64]

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @transpose() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      call @local_schedule_0(%c0, %c0) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }

  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %arg0, sizes: [128, 2, 64, 64], strides: [64, 8192, 16384, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1]>>
      %1 = ktdp.construct_memory_view %arg1, sizes: [128, 2, 64, 64], strides: [8192, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x2x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1]>> to memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1]>, "DDR">
      %cast = memref.cast %memspacecast : memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1]>, "DDR"> to memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<128x2x64x64xf16> to memref<128x2x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [128, 2, 64, 64], strides: [8192, 4096, 64, 1] : memref<128x2x64x64xf16, "DDR"> to memref<128x2x64x64xf16, strided<[8192, 4096, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<128x2x64x64xf16, strided<[8192, 4096, 64, 1]>, "DDR"> to memref<128x2x64x64xf16, strided<[8192, 4096, 64, 1], offset: ?>, "DDR">
      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %5 = arith.ceildivui %c128, %2 : index
      scf.for %arg2 = %c0 to %5 step %c1 {
        %6 = arith.ceildivui %c2, %3 : index
        %7 = ktdf.tiling.derive_size [%arg2 : %2], total_size = %c128 : index
        scf.for %arg3 = %c0 to %6 step %c1 {
          %8 = arith.ceildivui %c64, %4 : index
          %9 = ktdf.tiling.derive_size [%arg3 : %3], total_size = %c2 : index
          scf.for %arg4 = %c0 to %8 step %c1 {
            %10 = ktdf.tiling.derive_size [%arg4 : %4], total_size = %c64 : index
            scf.for %arg5 = %c0 to %7 step %c1 {
              %11 = ktdf.tiling.linearize_index [%arg2 : %2], [%arg5 : %c1] : index
              ktdf.pipeline {
                %12:4 = ktdf.private -> (memref<?x?x1x1x1x64xf16, "L1">, memref<?x?x1x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
                  %alloc = memref.alloc(%3, %4) : memref<?x?x1x1x1x64xf16, "L1">
                  %alloc_2 = memref.alloc(%3, %4) : memref<?x?x1x1x1x64xf16, "L1">
                  %13 = ktdf.create_token : !ktdf.token
                  %14 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %alloc, %alloc_2, %13, %14 : memref<?x?x1x1x1x64xf16, "L1">, memref<?x?x1x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%12#2) {
                  scf.for %arg6 = %c0 to %9 step %c1 {
                    %13 = ktdf.tiling.linearize_index [%arg3 : %3], [%arg6 : %c1] : index
                    scf.for %arg7 = %c0 to %10 step %c1 {
                      %14 = ktdf.tiling.linearize_index [%arg4 : %4], [%arg7 : %c1] : index
                      ktdf.data_transfer from %cast[%11, %13, %14, %c0 * 64] size [1, 1, 1, 64] to %12#0[%arg6, %arg7, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<128x2x64x64xf16, strided<[64, 8192, 16384, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
                    } {loop_type = #ktdf.loop_type<parallel_loop>}
                  } {loop_type = #ktdf.loop_type<parallel_loop>}
                } {applicable_units = ["MNILU"]}
                ktdf.stage depends_in(%12#2) depends_out(%12#3) {
                  scf.for %arg6 = %c0 to %9 step %c1 {
                    scf.for %arg7 = %c0 to %10 step %c1 {
                      ktdf.pipeline {
                        %13:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                          %14 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                          %15 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                          %16 = ktdf.create_token : !ktdf.token
                          %17 = ktdf.create_token : !ktdf.token
                          ktdf.private_yield %14, %15, %16, %17 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                        }
                        ktdf.stage depends_in(none) depends_out(%13#2) {
                          ktdf.data_transfer from %12#0[%arg6, %arg7, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %13#0 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                        } {applicable_units = ["L1LU"]}
                        ktdf.stage depends_in(%13#2) depends_out(%13#3) {
                          %14 = ktdf.read_from_fifo %13#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                          ktdf.write_to_fifo %14, %13#1 : tensor<1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                        } {applicable_units = ["SFU"]}
                        ktdf.stage depends_in(%13#3) depends_out(none) {
                          ktdf.data_transfer from %13#1 size [64] to %12#1[%arg6, %arg7, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x64xf16, "L1">
                        } {applicable_units = ["L1SU"]}
                      }
                    } {loop_type = #ktdf.loop_type<parallel_loop>}
                  } {loop_type = #ktdf.loop_type<parallel_loop>}
                } {applicable_units = ["L1LU", "SFU", "L1SU"]}
                ktdf.stage depends_in(%12#3) depends_out(none) {
                  scf.for %arg6 = %c0 to %9 step %c1 {
                    %13 = ktdf.tiling.linearize_index [%arg3 : %3], [%arg6 : %c1] : index
                    scf.for %arg7 = %c0 to %10 step %c1 {
                      %14 = ktdf.tiling.linearize_index [%arg4 : %4], [%arg7 : %c1] : index
                      ktdf.data_transfer from %12#1[%arg6, %arg7, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %cast_1[%11, %13, %14, %c0 * 64] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x64xf16, "L1">, memref<128x2x64x64xf16, strided<[8192, 4096, 64, 1], offset: ?>, "DDR">
                    } {loop_type = #ktdf.loop_type<parallel_loop>}
                  } {loop_type = #ktdf.loop_type<parallel_loop>}
                } {applicable_units = ["MNISU"]}
              }
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
