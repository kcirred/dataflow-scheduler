// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that ktdf.data_transfer with transfer_mode="lane0" produces:
//   1. dataflow.receive with the SOURCE FIFO slot size vector (not dest size)
//   2. vectorchain.shuffle taking lane 0 down to one element
//   3. agen.vector_store of that one lane at the destination indices, with a
//      one-element store_set

// CHECK:       #[[$SET:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0)>
// CHECK-LABEL: func.func @lane0_transfer
// CHECK:         dataflow.program_unit
// CHECK:           scf.for %[[ROW:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:        scf.for %[[COL:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:               %[[RECV:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK-NEXT:          %[[LANE:.+]] = vectorchain.shuffle input(%[[RECV]]) {indices = [0 : i32], repetition = 1 : i32} : vector<64xf16>, vector<1xf16>
// CHECK-NEXT:          agen.vector_store %[[LANE]], %{{.+}}[0, 0, %[[COL]], %[[ROW]]]
// CHECK-SAME:            store_set = #[[$SET]]
// CHECK-SAME:            : memref<1x1x64x64xf16, "L1">, vector<1xf16>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @lane0_transfer() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %map_l1su = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %u_l1su = uniform.query_map(map:%map_l1su, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
    %u_sfu  = uniform.query_map(map:%map_sfu,  key:%tile_id) : index
    ktdf_lowering.execute_on %u_sfu, %u_l1su {
      %alloc = memref.alloc() : memref<1x1x64x64xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %u_l1su {
        scf.for %row = %c0 to %c64 step %c1 {
          scf.for %col = %c0 to %c64 step %c1 {
            // Source size [64], dest size [1, 1, 1, 1]: store one lane
            ktdf.data_transfer from %fifo#0 size [64]
                               to %alloc[0, 0, %col, %row] size [1, 1, 1, 1]
                               {transfer_mode = "lane0"}
                : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x64x64xf16, "L1">
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}
