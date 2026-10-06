// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// PURPOSE: a non-square tile. The logical [128, 256] tensor, column-major in
// memory, is viewed as four 128x64 blocks: sizes [4, 128, 64], strides
// [8192, 1, 128] (one block = 8192 contiguous elements, element (row, c) at
// c * 128 + row). Checks that:
// - the source view is read as the row-major [128][64] tile it is in memory
//   (strides [8192, 64, 1]), not with its last two strides swapped;
// - the per-element re-stick splits output row t = 64 * t_hi + t_lo
//   (t_hi < 2, t_lo < 64) and loops c < 64, loading source column c from tile
//   element [2 * c + t_hi][t_lo] (its linear position c * 128 + t, with no
//   floordiv) and storing one element at [64 * t_hi + t_lo][c];
// - the row tile is pinned to all 128 rows.
// Catches PR #168's square-only loops (128 x 128 here, reading lanes past the
// 64-lane row) and its transposed [col, row] store, which assumes a square.

// CHECK-LABEL: func.func @local_schedule_0()
// CHECK:       ktdp.construct_memory_view %c64000, sizes: [4, 128, 64], strides: [8192, 64, 1]
// CHECK:       ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK:       ktdf.tiling.reserve_size {divisibility = 128 : index, min_value = 128 : index} : index

// MNILU: the whole block, verbatim.
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 128, 64] to %{{.*}} size [1, 1, 1, 128, 64]
// CHECK:       applicable_units = ["MNILU"]

// Middle stage: per element.
// CHECK:       scf.for %[[T_HI:.*]] = %{{.*}} to %c2 step
// CHECK:         scf.for %[[T_LO:.*]] = %{{.*}} to %c64{{[_0-9]*}} step
// CHECK:           scf.for %[[C:.*]] = %{{.*}} to %c64{{[_0-9]*}} step
// CHECK:             ktdf.data_transfer from %{{.*}}[%{{.*}}, %{{.*}}, 0, %[[C]] * 2 + %[[T_HI]], %[[T_LO]]] size [1, 1, 1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:             ktdf.data_transfer from %{{.*}} size [64] to %{{.*}}[%{{.*}}, %{{.*}}, 0, %[[T_HI]] * 64 + %[[T_LO]], %[[C]]] size [1, 1, 1, 1, 1] :

// MNISU: the whole block out.
// CHECK:       ktdf.data_transfer from %{{.*}} size [1, 1, 1, 128, 64] to %{{.*}} size [1, 128, 64]
// CHECK:       applicable_units = ["MNISU"]

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>

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
      %c128 = arith.constant 128 : index
      %c162304 = arith.constant 162304 : index
      %c64000 = arith.constant 64000 : index
      %c4 = arith.constant 4 : index

      %0 = ktdp.construct_memory_view %c64000, sizes: [4, 128, 64], strides: [8192, 1, 128] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128x64xf16, strided<[8192, 1, 128]>>
      %1 = ktdp.construct_memory_view %c162304, sizes: [4, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<4x128x64xf16, strided<[8192, 1, 128]>> to memref<4x128x64xf16, strided<[8192, 1, 128]>, "DDR">
      %cast = memref.cast %memspacecast : memref<4x128x64xf16, strided<[8192, 1, 128]>, "DDR"> to memref<4x128x64xf16, strided<[8192, 1, 128], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<4x128x64xf16> to memref<4x128x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [4, 128, 64], strides: [8192, 64, 1] : memref<4x128x64xf16, "DDR"> to memref<4x128x64xf16, strided<[8192, 64, 1]>, "DDR">
      %cast_1 = memref.cast %reinterpret_cast : memref<4x128x64xf16, strided<[8192, 64, 1]>, "DDR"> to memref<4x128x64xf16, strided<[8192, 64, 1], offset: ?>, "DDR">

      %2 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = arith.ceildivui %c4, %2 : index
      scf.for %arg0 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c128, %3 : index
        %6 = ktdf.tiling.derive_size [%arg0 : %2], total_size = %c4 : index
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
                  ktdf.data_transfer from %cast[%9, %10, %c0 * 64] size [1, 1, 64] to %8#0[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<4x128x64xf16, strided<[8192, 1, 128], offset: ?>, "DDR">, memref<?x?x1x1x64xf16, "L1">
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
                  ktdf.data_transfer from %8#1[%arg2, %arg3, 0, 0, 0] size [1, 1, 1, 1, 64] to %cast_1[%9, %10, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x64xf16, "L1">, memref<4x128x64xf16, strided<[8192, 64, 1], offset: ?>, "DDR">
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
