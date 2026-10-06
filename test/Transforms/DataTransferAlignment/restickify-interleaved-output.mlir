// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// PURPOSE: the destination is a [64, 128] row-major tensor seen as two 64x64
// blocks: sizes [2, 64, 64], strides [64, 128, 1]. Each stick is contiguous but
// the sticks of the two blocks interleave, so the strides do not decrease from
// the outermost dimension in. The device address generator derives each
// dimension's extent from the ratio of neighbouring strides and rejects that
// order, so DTA stores through a second view of the same memory with the
// dimensions reordered by stride ([64, 2, 64], strides [128, 64, 1]) and the
// store's subscripts and sizes permuted to match. Catches (a) a row-major
// rewrite of the destination to [4096, 64, 1], which silently stores the blocks
// one after the other, and (b) keeping the original order, which the device
// toolchain rejects ("Extent in load/store set is larger than from the
// layout"). Also pins the row tile to all 64 rows and the per-element re-stick
// between the two bulk copies.

// The reordered view's coordinate set swaps the bounds of d0 and d1.
// CHECK: #[[SET1:.*]] = affine_set<(d0, d1, d2) : (d1 >= 0, -d1 + 1 >= 0, d0 >= 0, -d0 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-LABEL: func.func @local_schedule_0()
// CHECK:       %[[REORDERED:.*]] = ktdp.construct_memory_view %c162304, sizes: [64, 2, 64], strides: [128, 64, 1] {coordinate_set = #[[SET1]], memory_space = #ktdp.memory_space<global>} : memref<64x2x64xf16, strided<[128, 64, 1]>>
// CHECK:       %[[BATCH_TILE:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:       %[[ROW_TILE:.*]] = ktdf.tiling.reserve_size {divisibility = 64 : index, min_value = 64 : index} : index

// Middle stage: per element, at the tile's own indices.
// CHECK:       ktdf.data_transfer from %{{.*}}[%[[B:arg[0-9]+]], %[[R:arg[0-9]+]], 0, %[[C:arg[0-9]+]], %[[T:arg[0-9]+]]] size [1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:       ktdf.data_transfer from %{{.*}} size [64] to %{{.*}}[%[[B]], %[[R]], 0, %[[T]], %[[C]]] size [1, 1, 1, 1, 1] :

// MNISU stage: one store per batch block, row index first.
// CHECK:       %[[BATCH:.*]] = ktdf.tiling.linearize_index [%{{.*}} : %[[BATCH_TILE]]], [%{{.*}} : %c1] : index
// CHECK:       %[[ROW:.*]] = ktdf.tiling.linearize_index [%{{.*}} : %[[ROW_TILE]]], [%{{.*}} : %c1] : index
// CHECK:       %[[MSC:.*]] = memref.memory_space_cast %[[REORDERED]] : memref<64x2x64xf16, strided<[128, 64, 1]>> to memref<64x2x64xf16, strided<[128, 64, 1]>, "DDR">
// CHECK:       %[[VIEW:.*]] = memref.cast %[[MSC]]
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 1, 64, 64] to %[[VIEW]][%[[ROW]], %[[BATCH]], %{{.*}}] size [64, 1, 64]
// CHECK-SAME:  memref<64x2x64xf16, strided<[128, 64, 1], offset: ?>, "DDR">
// CHECK:       applicable_units = ["MNISU"]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

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
      %c2 = arith.constant 2 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [2, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf16, strided<[4096, 1, 64]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [2, 64, 64], strides: [64, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf16, strided<[64, 128, 1]>>
      %memspacecast = memref.memory_space_cast %0 : memref<2x64x64xf16, strided<[4096, 1, 64]>> to memref<2x64x64xf16, strided<[4096, 1, 64]>, "DDR">
      %cast = memref.cast %memspacecast : memref<2x64x64xf16, strided<[4096, 1, 64]>, "DDR"> to memref<2x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64x64xf16, strided<[64, 128, 1]>> to memref<2x64x64xf16, strided<[64, 128, 1]>, "DDR">
      %cast_1 = memref.cast %memspacecast_0 : memref<2x64x64xf16, strided<[64, 128, 1]>, "DDR"> to memref<2x64x64xf16, strided<[64, 128, 1], offset: ?>, "DDR">

      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = arith.ceildivui %c2, %2 : index
      scf.for %arg0 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c64, %3 : index
        %6 = ktdf.tiling.derive_size [%arg0 : %2], total_size = %c2 : index
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
                  ktdf.data_transfer from %cast[%9, %10, %c0 * 64] size [1, 1, 64] to %8#0[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x64x64xf16, strided<[4096, 1, 64], offset: ?>, "DDR">, memref<?x?x1x1x64xf16, "L1">
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
                  ktdf.data_transfer from %8#1[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] to %cast_1[%9, %10, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x64xf16, "L1">, memref<2x64x64xf16, strided<[64, 128, 1], offset: ?>, "DDR">
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
