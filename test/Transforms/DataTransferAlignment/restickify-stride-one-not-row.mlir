// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// PURPOSE: a re-stickify whose input stride-1 dim is not next to the
// transfer's lane. transpose(1, 2) of a [128, 64, 64] fp16 tensor in
// torch-spyre's layout, over the shared index space (c, a, b) = [64, 128, 64]:
// input strides [1, 64, 8192] (stride 1 is dim 0), output strides
// [8192, 64, 1]. The two tile loops have different extents (64 for dim 0, 128
// for dim 1). Checks that the input view is rewritten into memory order
// ([8192, 64, 1]) and the load's sizes with it ([64, 1, 64]); that the tile of
// dim 0, the contiguous dim, is pinned to its 64 points while dim 1's tile
// stays free (128); and that the per-element re-stick and the bulk store
// [64, 1, 64] follow. Catches taking the lane's stride (8192) as the block
// length ("dimension total size (128) is not divisible by alignment factor
// 8192").

// CHECK-LABEL: func.func @local_schedule_0()
// CHECK:       ktdp.construct_memory_view %{{.*}}, sizes: [64, 128, 64], strides: [8192, 64, 1]
// CHECK:       ktdp.construct_memory_view %{{.*}}, sizes: [64, 128, 64], strides: [8192, 64, 1]
// CHECK:       %[[T0:.*]] = ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index} : index
// CHECK-NEXT:  %[[T1:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:       ktdf.tiling.derive_size [%{{.*}} : %[[T0]]], total_size = %c64 : index
// CHECK:       ktdf.tiling.derive_size [%{{.*}} : %[[T1]]], total_size = %c128 : index
// CHECK:       ktdf.data_transfer from %{{.*}} size [64, 1, 64] to %{{.*}} size [1, 1, 64, 1, 64]
// CHECK:       ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}, %[[R:arg[0-9]+]], 0, %[[C:arg[0-9]+]]] size [1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [64] to %{{.*}}[%{{.*}}, %{{.*}}, %[[C]], 0, %[[R]]] size [1, 1, 1, 1, 1] {transfer_mode = "lane0"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 64, 1, 64] to %{{.*}} size [64, 1, 64]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

module {
  module {
    func.func @transpose_0() attributes {grid = [6]} {
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
      %c1048576 = arith.constant 1048576 : index
      %c0_in = arith.constant 0 : index
      %c128 = arith.constant 128 : index

      %0 = ktdp.construct_memory_view %c0_in, sizes: [64, 128, 64], strides: [1, 64, 8192] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128x64xf16, strided<[1, 64, 8192]>>
      %1 = ktdp.construct_memory_view %c1048576, sizes: [64, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128x64xf16, strided<[8192, 64, 1]>>
      %memspacecast = memref.memory_space_cast %0 : memref<64x128x64xf16, strided<[1, 64, 8192]>> to memref<64x128x64xf16, strided<[1, 64, 8192]>, "DDR">
      %cast = memref.cast %memspacecast : memref<64x128x64xf16, strided<[1, 64, 8192]>, "DDR"> to memref<64x128x64xf16, strided<[1, 64, 8192], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<64x128x64xf16, strided<[8192, 64, 1]>> to memref<64x128x64xf16, strided<[8192, 64, 1]>, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [64, 128, 64], strides: [8192, 64, 1] : memref<64x128x64xf16, strided<[8192, 64, 1]>, "DDR"> to memref<64x128x64xf16, strided<[8192, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<64x128x64xf16, strided<[8192, 64, 1]>, "DDR"> to memref<64x128x64xf16, strided<[8192, 64, 1], offset: ?>, "DDR">

      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = arith.ceildivui %c64, %2 : index
      scf.for %arg0 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c128, %3 : index
        %6 = ktdf.tiling.derive_size [%arg0 : %2], total_size = %c64 : index
        scf.for %arg1 = %c0 to %5 step %c1 {
          %7 = ktdf.tiling.derive_size [%arg1 : %3], total_size = %c128 : index
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
                  ktdf.data_transfer from %cast[%9, %10, %c0 * 64] size [1, 1, 64] to %8#0[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x128x64xf16, strided<[1, 64, 8192], offset: ?>, "DDR">, memref<?x?x1x1x64xf16, "L1">
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
                  ktdf.data_transfer from %8#1[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] to %cast_1[%9, %10, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x64xf16, "L1">, memref<64x128x64xf16, strided<[8192, 64, 1], offset: ?>, "DDR">
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
