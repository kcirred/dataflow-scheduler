// RUN: dataflow-scheduler-opt --data-transfer-alignment %s -split-input-file | FileCheck %s

// PURPOSE: a copy whose x already has stride 1 last (strides [64, 1]) is not
// a re-stick: findRestick finds nothing to do, nothing is widened, and the
// views already decrease, so the module comes out unchanged.
// CHECK-LABEL: func.func @already_row_major
// CHECK-NOT:     transfer_mode
// CHECK:         memref.alloc() : memref<64x1x64xf16, "L1">
// CHECK:         ktdf.data_transfer from %{{.*}}[%{{.*}}, %c0 * 64] size [1, 64] to %{{.*}}[%{{.*}}, 0, 0] size [1, 1, 64]
// CHECK-NOT:     transfer_mode
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @already_row_major(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[64, 1]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16, strided<[64, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16> to memref<64x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<64x1x64xf16, "L1">
      %b = memref.alloc() : memref<64x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[64, 1]>, "DDR">, memref<64x1x64xf16, "L1">
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

// PURPOSE: orderViewsByStride lists a global view whose strides do not
// decrease ([64, 128, 1]) by decreasing stride ([128, 64, 1]), and moves
// everything that names its dims with it: sizes [2, 64, 64] -> [64, 2, 64],
// the coordinate set's bounds (dim 0 <= 1 now bounds d1), and the transfer's
// index slots and sizes (from [%i, %j, %c0] size [1, 2, 64] to
// [%j, %i, %c0] size [2, 1, 64]). Every element keeps its address. y already
// decreases and is untouched.
// CHECK:       #[[XSET:.*]] = affine_set<(d0, d1, d2) : (d1 >= 0, -d1 + 1 >= 0, d0 >= 0, -d0 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-LABEL: func.func @order
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:         ktdp.construct_memory_view %[[A]], sizes: [64, 2, 64], strides: [128, 64, 1] {coordinate_set = #[[XSET]]
// CHECK:         ktdp.construct_memory_view %[[B]], sizes: [2, 64, 64], strides: [4096, 64, 1]
// CHECK:         ktdf.data_transfer from %{{.*}}[%[[J]], %[[I]], %c0] size [2, 1, 64] to %{{.*}}[%[[I]], %[[J]], %c0] size [1, 2, 64]
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @order(%a: index, %b: index, %i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %x = ktdp.construct_memory_view %a, sizes: [2, 64, 64], strides: [64, 128, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf16, strided<[64, 128, 1]>>
  %y = ktdp.construct_memory_view %b, sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf16, strided<[4096, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<2x64x64xf16, strided<[64, 128, 1]>> to memref<2x64x64xf16, strided<[64, 128, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<2x64x64xf16, strided<[4096, 64, 1]>> to memref<2x64x64xf16, strided<[4096, 64, 1]>, "DDR">
  ktdf.data_transfer from %xd[%i, %j, %c0] size [1, 2, 64] to %yd[%i, %j, %c0] size [1, 2, 64] : memref<2x64x64xf16, strided<[64, 128, 1]>, "DDR">, memref<2x64x64xf16, strided<[4096, 64, 1]>, "DDR">
  return
}

// -----

// PURPOSE: a size-1 dim sorts first, whatever its stride (16384 here): its
// index is always 0, so its place does not change any address, and putting it
// first keeps the non-unit dims in decreasing order. x [2, 1, 64, 64] /
// [64, 16384, 128, 1] becomes [1, 64, 2, 64] / [16384, 128, 64, 1].
// CHECK-LABEL: func.func @unit_dim_first
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:         ktdp.construct_memory_view %[[A]], sizes: [1, 64, 2, 64], strides: [16384, 128, 64, 1]
// CHECK:         ktdf.data_transfer from %{{.*}}[%c0, %[[J]], %[[I]], %c0] size [1, 2, 1, 64] to %{{.*}}[%[[I]], %c0, %[[J]], %c0] size [1, 1, 2, 64]
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @unit_dim_first(%a: index, %b: index, %i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %x = ktdp.construct_memory_view %a, sizes: [2, 1, 64, 64], strides: [64, 16384, 128, 1] {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>} : memref<2x1x64x64xf16, strided<[64, 16384, 128, 1]>>
  %y = ktdp.construct_memory_view %b, sizes: [2, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>} : memref<2x1x64x64xf16, strided<[4096, 4096, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<2x1x64x64xf16, strided<[64, 16384, 128, 1]>> to memref<2x1x64x64xf16, strided<[64, 16384, 128, 1]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<2x1x64x64xf16, strided<[4096, 4096, 64, 1]>> to memref<2x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
  ktdf.data_transfer from %xd[%i, %c0, %j, %c0] size [1, 1, 2, 64] to %yd[%i, %c0, %j, %c0] size [1, 1, 2, 64] : memref<2x1x64x64xf16, strided<[64, 16384, 128, 1]>, "DDR">, memref<2x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
  return
}
