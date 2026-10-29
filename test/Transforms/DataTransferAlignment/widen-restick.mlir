// RUN: dataflow-scheduler-opt --data-transfer-alignment="order-views-by-stride=false" %s -split-input-file | FileCheck %s

// widenRestick, with the view reorder off so only the widening shows. Every
// input is a copy of the 2-D view x (strides [1, 64]: x's stick runs along dim
// 0, p = 0) into the row-major y (strides [64, 1]: y's stick runs along the
// last dim, 1), through staging buffers A and B, one stick per transfer.

// PURPOSE: the untiled copy is widened to one 64 x 64 tile.
//  - The point loop over dim p (%i, 64 trips) runs once in every stage.
//  - A and B change from 64x1x64 (one row per point) to 1x64x64 (one tile).
//  - The load reads x_t: x's memory with dims 0 and 1 swapped, strides
//    [64, 1], with its two slots swapped to match. Row a of A is then stick a
//    of x, 64 contiguous elements, which is what lowering reads when it moves
//    the last slot fastest. Reading x with its own strides [1, 64] would fill
//    row a with element a of every stick (a transpose of its own).
//  - x's view and cast have no other user and are erased.
//  - The nested pipeline is wrapped in loops r, c over [0, 64): it splats
//    A[point, r, c] and extracts into B[point, c, r], the only transpose.
//  - The store moves the whole tile to y in y's own slot order.
// CHECK-LABEL: func.func @untiled
// CHECK-NOT:     strides: [1, 64]
// CHECK:         %[[XT:.*]] = ktdp.construct_memory_view %arg0, sizes: [64, 64], strides: [64, 1]
// CHECK:         %[[XT_DDR:.*]] = memref.memory_space_cast %[[XT]] : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16, strided<[64, 1]>, "DDR">
// CHECK:         %[[BUF:.*]]:2 = ktdf.private -> (memref<1x64x64xf16, "L1">, memref<1x64x64xf16, "L1">)
// CHECK:         scf.for %[[I:.*]] = %c0 to %c1 step %c1 {
// CHECK-NEXT:      ktdf.data_transfer from %[[XT_DDR]][%c0 * 64, %[[I]]] size [64, 64] to %[[BUF]]#0[%[[I]], 0, 0] size [1, 64, 64]
// CHECK:         scf.for %[[I:.*]] = %c0 to %c1 step %c1 {
// CHECK-NEXT:      scf.for %[[R:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:        scf.for %[[C:.*]] = %c0 to %c64 step %c1 {
// CHECK-NEXT:          ktdf.pipeline {
// CHECK:                 ktdf.data_transfer from %[[BUF]]#0[%[[I]], %[[R]], %[[C]]] size [1, 1, 1] to %{{.*}} size [64] {transfer_mode = "splat"}
// CHECK:                 ktdf.data_transfer from %{{.*}} size [64] to %[[BUF]]#1[%[[I]], %[[C]], %[[R]]] size [1, 1, 1] {transfer_mode = "extract"}
// CHECK:         scf.for %[[I:.*]] = %c0 to %c1 step %c1 {
// CHECK-NEXT:      ktdf.data_transfer from %[[BUF]]#1[%[[I]], 0, 0] size [1, 64, 64] to %{{.*}}[%[[I]], %c0 * 64] size [64, 64]
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @untiled(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16> to memref<64x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<64x1x64xf16, "L1">
      %b = memref.alloc() : memref<64x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[1, 64]>, "DDR">, memref<64x1x64xf16, "L1">
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.pipeline {
          %f = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
            %s = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            ktdf.private_yield %s : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %buf#0[%i, 0, 0] size [1, 1, 64] to %f size [64] : memref<64x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %f size [64] to %buf#1[%i, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64x1x64xf16, "L1">
          }
        }
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: x keeps its view when something else reads it. The load reads the
// new x_t view; the other transfer still reads x through its own cast with
// strides [1, 64], unchanged. A pass that rewrote x in place would change what
// that transfer reads.
// CHECK-LABEL: func.func @x_has_other_user
// CHECK:         %[[X:.*]] = ktdp.construct_memory_view %arg0, sizes: [64, 64], strides: [1, 64]
// CHECK:         %[[XT:.*]] = ktdp.construct_memory_view %arg0, sizes: [64, 64], strides: [64, 1]
// CHECK:         %[[X_DDR:.*]] = memref.memory_space_cast %[[X]]
// CHECK:         %[[XT_DDR:.*]] = memref.memory_space_cast %[[XT]]
// CHECK:         ktdf.data_transfer from %[[X_DDR]][%c0, %c0] size [1, 1]
// CHECK:         ktdf.data_transfer from %[[XT_DDR]][%c0 * 64, %{{.*}}] size [64, 64]
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @x_has_other_user(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16> to memref<64x64xf16, "DDR">
  ktdf.data_transfer from %xd[%c0, %c0] size [1, 1] to %yd[%c0, %c0] size [1, 1] : memref<64x64xf16, strided<[1, 64]>, "DDR">, memref<64x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<64x1x64xf16, "L1">
      %b = memref.alloc() : memref<64x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[1, 64]>, "DDR">, memref<64x1x64xf16, "L1">
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.pipeline {
          %f = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
            %s = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            ktdf.private_yield %s : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %buf#0[%i, 0, 0] size [1, 1, 64] to %f size [64] : memref<64x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %f size [64] to %buf#1[%i, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64x1x64xf16, "L1">
          }
        }
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: a tiled loop over dim p (128 elements, tile size %tile from
// ktdf.tiling.reserve_size) gets one stick per tile. The placeholder becomes
// the constant 64, so the tile loop's trip count, derive_size and every
// linearize_index follow it; the point loop runs once; A and B lose their
// dynamic dim and hold one 1x64x64 tile. x (128 x 64, strides [1, 128]) is
// read through x_t (64 x 128, strides [128, 1]).
// CHECK-LABEL: func.func @tiled
// CHECK-NOT:     ktdf.tiling.reserve_size
// CHECK:         %[[XT:.*]] = ktdp.construct_memory_view %arg0, sizes: [64, 128], strides: [128, 1]
// CHECK:         %[[XT_DDR:.*]] = memref.memory_space_cast %[[XT]]
// CHECK:         %[[NT:.*]] = arith.ceildivui %c128, %c64 : index
// CHECK:         scf.for %[[T:.*]] = %c0 to %[[NT]] step %c1 {
// CHECK:           ktdf.tiling.derive_size [%[[T]] : %c64], total_size = %c128 : index
// CHECK:           %[[BUF:.*]]:2 = ktdf.private -> (memref<1x64x64xf16, "L1">, memref<1x64x64xf16, "L1">)
// CHECK:             memref.alloc() : memref<1x64x64xf16, "L1">
// CHECK:           scf.for %[[I:.*]] = %c0 to %c1 step %c1 {
// CHECK-NEXT:        %[[ROW:.*]] = ktdf.tiling.linearize_index [%[[T]] : %c64], [%[[I]] : %c1] : index
// CHECK-NEXT:        ktdf.data_transfer from %[[XT_DDR]][%c0 * 64, %[[ROW]]] size [64, 64] to %[[BUF]]#0[%[[I]], 0, 0] size [1, 64, 64]
// CHECK:           ktdf.tiling.linearize_index [%[[T]] : %c64]
// CHECK-NEXT:        ktdf.data_transfer from %[[BUF]]#1[%{{.*}}, 0, 0] size [1, 64, 64] to %{{.*}}[%{{.*}}, %c0 * 64] size [64, 64]
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @tiled(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [128, 64], strides: [1, 128] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16, strided<[1, 128]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [128, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %xd = memref.memory_space_cast %x : memref<128x64xf16, strided<[1, 128]>> to memref<128x64xf16, strided<[1, 128]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<128x64xf16> to memref<128x64xf16, "DDR">
  %tile = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
  %ntiles = arith.ceildivui %c128, %tile : index
  scf.for %t = %c0 to %ntiles step %c1 {
    %n = ktdf.tiling.derive_size [%t : %tile], total_size = %c128 : index
    ktdf.pipeline {
      %buf:2 = ktdf.private -> (memref<?x1x64xf16, "L1">, memref<?x1x64xf16, "L1">) {
        %a = memref.alloc(%tile) : memref<?x1x64xf16, "L1">
        %b = memref.alloc(%tile) : memref<?x1x64xf16, "L1">
        ktdf.private_yield %a, %b : memref<?x1x64xf16, "L1">, memref<?x1x64xf16, "L1">
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          %row = ktdf.tiling.linearize_index [%t : %tile], [%i : %c1] : index
          ktdf.data_transfer from %xd[%row, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<128x64xf16, strided<[1, 128]>, "DDR">, memref<?x1x64xf16, "L1">
        }
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          ktdf.pipeline {
            %f = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
              %s = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
              ktdf.private_yield %s : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            }
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %buf#0[%i, 0, 0] size [1, 1, 64] to %f size [64] : memref<?x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            }
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %f size [64] to %buf#1[%i, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<?x1x64xf16, "L1">
            }
          }
        }
      }
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %n step %c1 {
          %row = ktdf.tiling.linearize_index [%t : %tile], [%i : %c1] : index
          ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%row, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x1x64xf16, "L1">, memref<128x64xf16, "DDR">
        }
      }
    }
  }
  return
}
