// RUN: dataflow-scheduler-opt --data-transfer-alignment %s -split-input-file -verify-diagnostics

// findRestick, one rejected input per case. Each is the same 64x64 copy
// x -> A -> B -> y (x strides [1, 64], y row-major [64, 1], one stick per
// transfer) with one thing changed. Once the load from x into the scratchpad
// shows x's last stride is not 1 the copy is a re-stick, and every unmet
// expectation must be an error naming it, never a silent skip.

// PURPOSE: findRestick rejects the copy: x's last stride is not 1, so this is a
// re-stick, but no dim of x has stride 1 (strides [2, 64]): there is no stick
// of x to read whole.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @unitStrideDim(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [2, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[2, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[2, 64]>> to memref<64x64xf16, strided<[2, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16> to memref<64x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<64x1x64xf16, "L1">
      %b = memref.alloc() : memref<64x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<64x1x64xf16, "L1">, memref<64x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c64 step %c1 {
        // expected-error @+1 {{the source view has no dim with stride 1}}
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[2, 64]>, "DDR">, memref<64x1x64xf16, "L1">
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

// PURPOSE: findRestick rejects the copy: y must be row-major (its last dim has
// stride 1): y here has strides [1, 64].
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @isRowMajor(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
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
        // expected-error @+1 {{the destination view is not row-major}}
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, strided<[1, 64]>, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: x and y must list the same dims: x is
// rank 2, y rank 3.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @rank(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 1, 64], strides: [64, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x1x64xf16, strided<[64, 64, 1]>>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x1x64xf16, strided<[64, 64, 1]>> to memref<64x1x64xf16, strided<[64, 64, 1]>, "DDR">
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
        // expected-error @+1 {{the source and destination views differ in rank (2 vs 3)}}
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x1x64xf16, strided<[64, 64, 1]>, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: x's stride-1 dim (dim 0, extent 96)
// must be whole sticks of 64.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @extent(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [96, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
  %xd = memref.memory_space_cast %x : memref<96x64xf16, strided<[1, 64]>> to memref<96x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<96x64xf16> to memref<96x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<96x1x64xf16, "L1">, memref<96x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<96x1x64xf16, "L1">
      %b = memref.alloc() : memref<96x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<96x1x64xf16, "L1">, memref<96x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c96 step %c1 {
        // expected-error @+1 {{dim 0 has extent 96, not a multiple of the stick size 64}}
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<96x64xf16, strided<[1, 64]>, "DDR">, memref<96x1x64xf16, "L1">
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c96 step %c1 {
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<96x1x64xf16, "L1">, memref<96x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: each global-side transfer must move
// one stick: sizes [1, 64]. This load moves [2, 64].
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @sizes(%x_addr: index, %y_addr: index) {
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
        // expected-error @+1 {{global-side size in slot 0 is not 1; the pass expects one stick per transfer}}
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [2, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[1, 64]>, "DDR">, memref<64x1x64xf16, "L1">
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

// PURPOSE: findRestick rejects the copy: the store must address the same
// element as the load in every slot: the load uses %i in slot 0, the store %c0,
// so the tile read from x is not the tile written to y.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @index(%x_addr: index, %y_addr: index) {
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
        // expected-error @+1 {{slot 0 of the store to y does not use the same index as slot 0 of the load from x}}
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%c0, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: the A -> B stage must be a nested
// ktdf.pipeline; here there is none.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @nested(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [64, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %xd = memref.memory_space_cast %x : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<64x64xf16> to memref<64x64xf16, "DDR">
  // expected-error @+1 {{no nested pipeline moves sticks from A to B}}
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
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: slot p (0) of the load must be indexed
// by the loop over dim p; here it is the constant %c0 (the store matches it, so
// the index check passes).
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @slotp(%x_addr: index, %y_addr: index) {
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
        // expected-error @+1 {{slot 0 of the load is not indexed by a loop over dim 0}}
        ktdf.data_transfer from %xd[%c0, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x64xf16, strided<[1, 64]>, "DDR">, memref<64x1x64xf16, "L1">
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
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%c0, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<64x1x64xf16, "L1">, memref<64x64xf16, "DDR">
      }
    }
  }
  return
}

// -----

// PURPOSE: findRestick rejects the copy: an untiled loop over dim p must cover
// exactly one stick; here it covers 128 elements.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @untiled(%x_addr: index, %y_addr: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %x = ktdp.construct_memory_view %x_addr, sizes: [128, 64], strides: [1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16, strided<[1, 64]>>
  %y = ktdp.construct_memory_view %y_addr, sizes: [128, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %xd = memref.memory_space_cast %x : memref<128x64xf16, strided<[1, 64]>> to memref<128x64xf16, strided<[1, 64]>, "DDR">
  %yd = memref.memory_space_cast %y : memref<128x64xf16> to memref<128x64xf16, "DDR">
  ktdf.pipeline {
    %buf:2 = ktdf.private -> (memref<128x1x64xf16, "L1">, memref<128x1x64xf16, "L1">) {
      %a = memref.alloc() : memref<128x1x64xf16, "L1">
      %b = memref.alloc() : memref<128x1x64xf16, "L1">
      ktdf.private_yield %a, %b : memref<128x1x64xf16, "L1">, memref<128x1x64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c128 step %c1 {
        // expected-error @+1 {{dim 0 is walked by an untiled loop over 128 elements; only one stick (64) is supported}}
        ktdf.data_transfer from %xd[%i, %c0 * 64] size [1, 64] to %buf#0[%i, 0, 0] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<128x64xf16, strided<[1, 64]>, "DDR">, memref<128x1x64xf16, "L1">
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c128 step %c1 {
        ktdf.pipeline {
          %f = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
            %s = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
            ktdf.private_yield %s : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %buf#0[%i, 0, 0] size [1, 1, 64] to %f size [64] : memref<128x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
          }
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %f size [64] to %buf#1[%i, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<128x1x64xf16, "L1">
          }
        }
      }
    }
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %c128 step %c1 {
        ktdf.data_transfer from %buf#1[%i, 0, 0] size [1, 1, 64] to %yd[%i, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<128x1x64xf16, "L1">, memref<128x64xf16, "DDR">
      }
    }
  }
  return
}
