// RUN: dataflow-scheduler-opt --data-transfer-alignment %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: dataflow-scheduler-opt --data-transfer-alignment --ktdf-to-ktdflowering --ktdflowering-to-dfir %s -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=DFIR

// XFAIL: *
// Pins a bug: orderViewsByStride permutes only the reordered view's side of a
// transfer, so a transfer that walks the view in two slots pairs the wrong
// elements, and the two error cases below go unreported. Fixed looks like:
// reorderView also reorders the staging buffer on the other side when the
// walk changes (CHECKs in @buffer_follows), and reports the two errors. Then
// this test passes unexpectedly; remove this XFAIL with that fix.

// orderViewsByStride, when a transfer on the reordered view moves more than
// one element in two or more slots. A ktdf.data_transfer pairs elements in
// walk order: lowering steps through each side's slots of size other than 1
// in slot order, and the k-th element read is the k-th element written.
// Reordering the view's slots on one side alone changes that side's walk and
// so which element lands where. The other side must be reordered the same
// way, or the pass must stop.

// PURPOSE: the other side is a staging buffer, so it follows the view.
// x is 2 x 3 sticks with strides [64, 128, 1]: x[i][j] is at 64*i + 128*j.
// The load moves all of x into A (size [2, 3, 64]); a second stage copies
// A[i][j] to y[i][j]. So y[i][j] = x[i][j].
// orderViewsByStride lists x as [3, 2, 64] / [128, 64, 1]. The load now walks
// x's old dim 1 first. A is reordered with it: its dims become [3, 2, 64], the
// load's A side is resized to match, and the reader indexes A as [%j, %i].
// So A'[a][b] = x[b][a] and y[i][j] = A'[j][i] = x[i][j], as before.
// Reordering x alone left A as 2 x 3: the load then paired x's 3-step walk
// with A's 2-step walk. Lowering rejected that for these extents; for equal
// extents (2 x 2) it lowered, and A[0][1] and A[1][0] swapped rows.
// CHECK-LABEL: func.func @buffer_follows
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [3, 2, 64], strides: [128, 64, 1]
// CHECK:         %[[BUF:.*]] = ktdf.private -> (memref<3x2x64xf16, "L1">)
// CHECK:           memref.alloc() : memref<3x2x64xf16, "L1">
// CHECK:         ktdf.data_transfer from %{{.*}}[%c0, %c0, %c0] size [3, 2, 64] to %[[BUF]][0, 0, 0] size [3, 2, 64]
// CHECK:         scf.for %[[I:.*]] = %c0 to %c2
// CHECK:           scf.for %[[J:.*]] = %c0 to %c3
// CHECK:             ktdf.data_transfer from %[[BUF]][%[[J]], %[[I]], 0] size [1, 1, 64] to %{{.*}}[%[[I]], %[[J]], %c0] size [1, 1, 64]
// The load lowers with both sides walking the same distance per step: x's
// layout and A's row-major 3x2x64 both advance 128 then 64 elements.
// DFIR-LABEL: func.func @buffer_follows
// DFIR:         dataflow.get_logical_memory_view %{{.*}} {layout_map = #[[XMAP:.*]]} : index, index, memref<3x2x64xf16>
// DFIR:         agen.composite_load_and_store src:%{{.*}}[%c0, %c0, %c0] dst:%{{.*}}[0, 0, 0]
// DFIR:         } : memref<3x2x64xf16>, memref<3x2x64xf16, "L1">

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 2 >= 0, d2 >= 0, -d2 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @buffer_follows(%x_addr: index, %y_addr: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [2, 3, 64], strides: [64, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[64, 128, 1]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [2, 3, 64], strides: [192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[192, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<2x3x64xf16, strided<[64, 128, 1]>> to memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<2x3x64xf16, strided<[192, 64, 1]>> to memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
  ktdf.pipeline {
    %buf = ktdf.private -> (memref<2x3x64xf16, "L1">) {
      %a = memref.alloc() : memref<2x3x64xf16, "L1">
      ktdf.private_yield %a : memref<2x3x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %xd[%c0, %c0, %c0] size [2, 3, 64] to %buf[0, 0, 0] size [2, 3, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">, memref<2x3x64xf16, "L1">
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c3 step %c1 {
          ktdf.data_transfer from %buf[%i, %j, 0] size [1, 1, 64] to %yd[%i, %j, %c0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x3x64xf16, "L1">, memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
        }
      }
    } {applicable_units = ["MNISU"]}
  }
  return
}

// -----

// PURPOSE: the other side is a global view, which the pass does not reorder
// to suit x (y is row-major and stays so). No slot order serves both walks,
// so the pass must report it rather than change the pairing.

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 2 >= 0, d2 >= 0, -d2 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @global_to_global(%x_addr: index, %y_addr: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [2, 3, 64], strides: [64, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[64, 128, 1]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [2, 3, 64], strides: [192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[192, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<2x3x64xf16, strided<[64, 128, 1]>> to memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<2x3x64xf16, strided<[192, 64, 1]>> to memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
  // expected-error @+1 {{reordering the source view by stride changes the order this transfer walks it, and the destination is not a staging buffer that can be reordered to match}}
  ktdf.data_transfer from %xd[%c0, %c0, %c0] size [2, 3, 64] to %yd[%c0, %c0, %c0] size [2, 3, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">, memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
  return
}

// -----

// PURPOSE: the staging buffer has a second transfer that walks it in two
// slots. Reordering A for the load would change that transfer's walk too,
// and its other side (y) cannot follow. The pass must stop rather than fix one
// pairing by breaking another.

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 2 >= 0, d2 >= 0, -d2 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @buffer_shared(%x_addr: index, %y_addr: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [2, 3, 64], strides: [64, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[64, 128, 1]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [2, 3, 64], strides: [192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16, strided<[192, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<2x3x64xf16, strided<[64, 128, 1]>> to memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<2x3x64xf16, strided<[192, 64, 1]>> to memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
  ktdf.pipeline {
    %buf = ktdf.private -> (memref<2x3x64xf16, "L1">) {
      %a = memref.alloc() : memref<2x3x64xf16, "L1">
      ktdf.private_yield %a : memref<2x3x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %xd[%c0, %c0, %c0] size [2, 3, 64] to %buf[0, 0, 0] size [2, 3, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x3x64xf16, strided<[64, 128, 1]>, "DDR">, memref<2x3x64xf16, "L1">
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @+1 {{reordering a staging buffer to follow a view would change the order this transfer walks the buffer}}
      ktdf.data_transfer from %buf[0, 0, 0] size [2, 3, 64] to %yd[%c0, %c0, %c0] size [2, 3, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<2x3x64xf16, "L1">, memref<2x3x64xf16, strided<[192, 64, 1]>, "DDR">
    } {applicable_units = ["MNISU"]}
  }
  return
}
