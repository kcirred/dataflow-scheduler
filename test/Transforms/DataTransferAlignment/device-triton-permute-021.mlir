// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// PURPOSE: Triton's tl.permute(x, (0, 2, 1)) of a [64, 64, 64] fp16 tensor in
// torch-spyre's layout. The input's stride-1 dim is next to the transfer's
// lane, but its columns are 4096 apart (strides [.., 64, 1, 4096]). Checks
// that the input view is rewritten into memory order ([.., 4096, 64, 1]),
// that the tile of the contiguous dim is pinned to its 64 points, and that the
// re-stick follows: bulk load [1, 1, 64, 1, 64], one broadcast load and one
// lane-0 store per element with the two element indices exchanged, bulk store
// [1, 1, 64, 1, 64]. Catches taking the lane's stride (4096) as the block
// length ("dimension total size (64) is not divisible by alignment factor
// 4096").
//
// The input is the scheduler IR that DataTransferAlignment received for a
// kernel that ran correctly on Spyre DD2 (exact compare against numpy), printed
// with --mlir-print-ir-before=data-transfer-alignment. It is retargeted at the
// test device restickify_device.mlir, which has DD2's scratchpad access sizes:
// only the device line and the unit names changed (L3LU -> MNILU, LXLU -> L1LU,
// SFP -> SFU, LXSU -> L1SU, L3SU -> MNISU). DataTransferAlignment's output on
// this device matches its output on DD2 apart from those names.

// CHECK-LABEL: func.func @local_schedule_0(
// CHECK:       ktdp.construct_memory_view %arg0, sizes: [1, 1, 64, 64, 64], strides: [262144, 262144, 4096, 64, 1]
// CHECK:       ktdp.construct_memory_view %arg1, sizes: [1, 1, 64, 64, 64], strides: [262144, 4096, 4096, 64, 1]
// CHECK:       ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:  ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index} : index
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 64, 1, 64] to %{{.*}} size [1, 1, 1, 1, 1, 64, 64]
// CHECK:       ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}, 0, 0, 0, %[[R:arg[0-9]+]], %[[C:arg[0-9]+]]] size [1, 1, 1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [64] to %{{.*}}[%{{.*}}, %{{.*}}, 0, 0, 0, %[[C]], %[[R]]] size [1, 1, 1, 1, 1, 1, 1] {transfer_mode = "lane0"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 1, 1, 1, 64, 64] to %{{.*}} size [1, 1, 64, 1, 64]

#set = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  module {
    func.func @permute3d_stick(%arg0: index, %arg1: index) attributes {grid = [1]} {
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
      %0 = ktdp.construct_memory_view %arg0, sizes: [1, 1, 64, 64, 64], strides: [4096, 262144, 64, 1, 4096] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096]>>
      %1 = ktdp.construct_memory_view %arg1, sizes: [1, 1, 64, 64, 64], strides: [262144, 4096, 64, 4096, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1]>>
      %memspacecast = memref.memory_space_cast %0 : memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096]>> to memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096]>, #ktdp.memory_space<global>>
      %cast = memref.cast %memspacecast : memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096]>, #ktdp.memory_space<global>> to memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1]>> to memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1]>, #ktdp.memory_space<global>>
      %cast_1 = memref.cast %memspacecast_0 : memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1]>, #ktdp.memory_space<global>> to memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1], offset: ?>, #ktdp.memory_space<global>>
      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = arith.ceildivui %c64, %2 : index
      scf.for %arg2 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c64, %3 : index
        %6 = ktdf.tiling.derive_size [%arg2 : %2], total_size = %c64 : index
        scf.for %arg3 = %c0 to %5 step %c1 {
          %7 = ktdf.tiling.derive_size [%arg3 : %3], total_size = %c64 : index
          ktdf.pipeline {
            %8:4 = ktdf.private -> (memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
              %alloc = memref.alloc(%2, %3) : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
              %alloc_2 = memref.alloc(%2, %3) : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
              %9 = ktdf.create_token : !ktdf.token
              %10 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %alloc, %alloc_2, %9, %10 : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%8#2) {
              scf.for %arg4 = %c0 to %6 step %c1 {
                %9 = ktdf.tiling.linearize_index [%arg2 : %2], [%arg4 : %c1] : index
                scf.for %arg5 = %c0 to %7 step %c1 {
                  %10 = ktdf.tiling.linearize_index [%arg3 : %3], [%arg5 : %c1] : index
                  ktdf.data_transfer from %cast[%c0, %c0, %9, %10, %c0 * 64] size [1, 1, 1, 1, 64] to %8#0[%arg4, %arg5, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x1x64x64x64xf16, strided<[4096, 262144, 64, 1, 4096], offset: ?>, #ktdp.memory_space<global>>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["MNILU"]}
            ktdf.stage depends_in(%8#2) depends_out(%8#3) {
              scf.for %arg4 = %c0 to %6 step %c1 {
                scf.for %arg5 = %c0 to %7 step %c1 {
                  ktdf.pipeline {
                    %9:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                      %10 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      %11 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                      %12 = ktdf.create_token : !ktdf.token
                      %13 = ktdf.create_token : !ktdf.token
                      ktdf.private_yield %10, %11, %12, %13 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                    }
                    ktdf.stage depends_in(none) depends_out(%9#2) {
                      ktdf.data_transfer from %8#0[%arg4, %arg5, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %9#0 size [64] : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    } {applicable_units = ["L1LU"]}
                    ktdf.stage depends_in(%9#2) depends_out(%9#3) {
                      %10 = ktdf.read_from_fifo %9#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x1x64xf16>
                      ktdf.write_to_fifo %10, %9#1 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    } {applicable_units = ["SFU"]}
                    ktdf.stage depends_in(%9#3) depends_out(none) {
                      ktdf.data_transfer from %9#1 size [64] to %8#1[%arg4, %arg5, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>
                    } {applicable_units = ["L1SU"]}
                  }
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["L1LU", "SFU", "L1SU"]}
            ktdf.stage depends_in(%8#3) depends_out(none) {
              scf.for %arg4 = %c0 to %6 step %c1 {
                %9 = ktdf.tiling.linearize_index [%arg2 : %2], [%arg4 : %c1] : index
                scf.for %arg5 = %c0 to %7 step %c1 {
                  %10 = ktdf.tiling.linearize_index [%arg3 : %3], [%arg5 : %c1] : index
                  ktdf.data_transfer from %8#1[%arg4, %arg5, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %cast_1[%c0, %c0, %9, %10, %c0 * 64] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x1x64xf16, #ktdp.memory_space<ct_local>>, memref<1x1x64x64x64xf16, strided<[262144, 4096, 64, 4096, 1], offset: ?>, #ktdp.memory_space<global>>
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
