// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// PURPOSE: torch-spyre's KTIR for torch.transpose(x, 1, 2).contiguous() of a
// [128, 64, 128] fp16 tensor, over the shared index space
// [1, c // 64, c % 64, a, b] = [1, 2, 64, 128, 64]. The input's stride-1 dim
// (c % 64) is at position 2, not next to the transfer's lane (b). Checks that
// the input view is rewritten into memory order ([1, 64, 2, 128, 64], strides
// decreasing), that the tile of c % 64 is pinned to its 64 points, and that
// the re-stick follows: bulk load [1, 64, 1, 1, 64], one broadcast load and
// one lane-0 store per element with the two element indices exchanged, and
// the bulk store [1, 1, 64, 1, 64] into the output view, already in memory
// order. Catches taking b's stride (16384) as the block length ("dimension
// total size (128) is not divisible by alignment factor 16384").
//
// The input is the scheduler IR that DataTransferAlignment received for a
// kernel that ran correctly on Spyre DD2 (exact compare against numpy), printed
// with --mlir-print-ir-before=data-transfer-alignment. It is retargeted at the
// test device restickify_device.mlir, which has DD2's scratchpad access sizes:
// only the device line and the unit names changed (L3LU -> MNILU, LXLU -> L1LU,
// SFP -> SFU, LXSU -> L1SU, L3SU -> MNISU). DataTransferAlignment's output on
// this device matches its output on DD2 apart from those names.

// CHECK-LABEL: func.func @local_schedule_0(
// CHECK:       ktdp.construct_memory_view %arg0, sizes: [1, 64, 2, 128, 64], strides: [1048576, 16384, 8192, 64, 1]
// CHECK:       ktdp.construct_memory_view %arg1, sizes: [1, 2, 64, 128, 64], strides: [1048576, 524288, 8192, 64, 1]
// CHECK:       ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:  %[[TP:.*]] = ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index} : index
// CHECK-NEXT:  ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:       ktdf.tiling.derive_size [%{{.*}} : %[[TP]]], total_size = %c64 : index
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 64, 1, 1, 64] to %{{.*}} size [1, 1, 1, 1, 64, 1, 64]
// CHECK:       ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}, 0, 0, %[[R:arg[0-9]+]], 0, %[[C:arg[0-9]+]]] size [1, 1, 1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [64] to %{{.*}}[%{{.*}}, %{{.*}}, 0, 0, %[[C]], 0, %[[R]]] size [1, 1, 1, 1, 1, 1, 1] {transfer_mode = "lane0"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 1, 1, 64, 1, 64] to %{{.*}} size [1, 1, 64, 1, 64]

#set = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 127 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  module {
    func.func @ktir_fused_clone_transpose_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/restickify_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c128 = arith.constant 128 : index
      %0 = ktdp.construct_memory_view %arg0, sizes: [1, 2, 64, 128, 64], strides: [1048576, 8192, 1, 64, 16384] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384]>>
      %1 = ktdp.construct_memory_view %arg1, sizes: [1, 2, 64, 128, 64], strides: [1048576, 524288, 8192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x2x64x128x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384]>> to memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384]>, #ktdp.memory_space<global>>
      %cast = memref.cast %memspacecast : memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384]>, #ktdp.memory_space<global>> to memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x2x64x128x64xf16> to memref<1x2x64x128x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 2, 64, 128, 64], strides: [1048576, 524288, 8192, 64, 1] : memref<1x2x64x128x64xf16, #ktdp.memory_space<global>> to memref<1x2x64x128x64xf16, strided<[1048576, 524288, 8192, 64, 1]>, #ktdp.memory_space<global>>
      %cast_1 = memref.cast %reinterpret_cast : memref<1x2x64x128x64xf16, strided<[1048576, 524288, 8192, 64, 1]>, #ktdp.memory_space<global>> to memref<1x2x64x128x64xf16, strided<[1048576, 524288, 8192, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %5 = arith.ceildivui %c2, %2 : index
      scf.for %arg2 = %c0 to %5 step %c1 {
        %6 = arith.ceildivui %c64, %3 : index
        %7 = ktdf.tiling.derive_size [%arg2 : %2], total_size = %c2 : index
        scf.for %arg3 = %c0 to %6 step %c1 {
          %8 = arith.ceildivui %c128, %4 : index
          %9 = ktdf.tiling.derive_size [%arg3 : %3], total_size = %c64 : index
          scf.for %arg4 = %c0 to %8 step %c1 {
            %10 = ktdf.tiling.derive_size [%arg4 : %4], total_size = %c128 : index
            scf.for %arg5 = %c0 to %7 step %c1 {
              %11 = ktdf.tiling.linearize_index [%arg2 : %2], [%arg5 : %c1] : index
              ktdf.pipeline {
                %12:4 = ktdf.private -> (memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
                  %alloc = memref.alloc(%3, %4) : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
                  %alloc_2 = memref.alloc(%3, %4) : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
                  %13 = ktdf.create_token : !ktdf.token
                  %14 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %alloc, %alloc_2, %13, %14 : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%12#2) {
                  scf.for %arg6 = %c0 to %9 step %c1 {
                    %13 = ktdf.tiling.linearize_index [%arg3 : %3], [%arg6 : %c1] : index
                    scf.for %arg7 = %c0 to %10 step %c1 {
                      %14 = ktdf.tiling.linearize_index [%arg4 : %4], [%arg7 : %c1] : index
                      ktdf.data_transfer from %cast[%c0, %11, %13, %14, %c0 * 64] size [1, 1, 1, 1, 64] to %12#0[%arg6, %arg7, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x2x64x128x64xf16, strided<[1048576, 8192, 1, 64, 16384], offset: ?>, #ktdp.memory_space<global>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
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
                          ktdf.data_transfer from %12#0[%arg6, %arg7, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %13#0 size [64] : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                        } {applicable_units = ["L1LU"]}
                        ktdf.stage depends_in(%13#2) depends_out(%13#3) {
                          %14 = ktdf.read_from_fifo %13#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x1x64xf16>
                          ktdf.write_to_fifo %14, %13#1 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                        } {applicable_units = ["SFU"]}
                        ktdf.stage depends_in(%13#3) depends_out(none) {
                          ktdf.data_transfer from %13#1 size [64] to %12#1[%arg6, %arg7, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
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
                      ktdf.data_transfer from %12#1[%arg6, %arg7, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %cast_1[%c0, %11, %13, %14, %c0 * 64] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<1x2x64x128x64xf16, strided<[1048576, 524288, 8192, 64, 1], offset: ?>, #ktdp.memory_space<global>>
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
