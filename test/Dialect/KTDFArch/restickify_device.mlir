// RUN: dataflow-scheduler-opt %s

// The sample device with the load/store geometry of Spyre DD2's LX units:
// 128-byte (a stick of 64 f16), 16-byte (a slice of 8 f16) and 2-byte (one f16)
// accesses, and 64 f16 / 32 f32 SIMD lanes. Used by the DataTransferAlignment
// re-stickify tests, whose element broadcast and one-element store depend on
// the 2-byte access.

#DDR = {
  kind = "DDR",
  size = 1073741824 //1GB
}
#DDR_MNILU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU_DDR = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#L1 = {
  kind = "L1",
  size = 1048576 //1MB
}
#IAB = {
  kind = "IAB",
  ktdf_arch.features = {
    ktdf_arch.feature.indirect_address_buffer = { num_entries = 32, entry_type = si32 }
  }
}
#MNILU = {
  kind = "MNILU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
  dataflow_scheduler.double_buffer_last
}
#MNILU_L1 = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU = {
  kind = "MNISU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
  dataflow_scheduler.double_buffer_last
}
#L1_MNISU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#SFU = {
  kind = "SFU",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 64, f32 = 32> }
  }
}

#SFU_REG = {
  kind = "SFU_REG",
  size = 2048 //2KB
}

#L1LU = {
  kind = "L1LU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 128, align_in_words = 128},
          {size_in_words = 16, align_in_words = 16},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    },
    ktdf_arch.feature.simd = { splat, zero_pad }
  }
}
#L1LU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1LU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 128, 16, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

#L1SU = {
  kind = "L1SU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 128, align_in_words = 128},
          {size_in_words = 16, align_in_words = 16},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    }
  }
}
#L1SU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1SU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 128, 16, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

ktdf_arch.device @sample_device {
  %ddr = memory #DDR

  // First core.
  group { kind = "core" } share(%ddr) {
    // Private scratchpad memory.
    %l1 = memory #L1

    // DDR <-> L1 DMA units (each wraps an IAB co-located with the unit).
    %mnilu = group { kind = "MNILU_Block" } share() {
      %iab = memory #IAB
      %mnilu = exec_unit #MNILU
      yield %mnilu
    } -> exec_unit

    %mnisu = group { kind = "MNISU_Block" } share() {
      %iab = memory #IAB
      %mnisu = exec_unit #MNISU
      yield %mnisu
    } -> exec_unit

    // DMA datapaths: 256B/cy read, 128B/cy write.
    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    // First sub-core.
    group  share(%l1) {
      // Compute units.
      %sfu = group { kind = "SFU_Block" } share() {
         %sfp_reg = memory #SFU_REG
         %sfp_unit = exec_unit #SFU
         yield %sfp_unit
      } -> exec_unit

      // L1 Load/Store units.
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      // FIFO datapaths.
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }

    // Second sub-core.
    group  share(%l1) {
      // Compute units.
      %sfu = group { kind = "SFU_Block" } share() {
         %sfp_reg = memory #SFU_REG
         %sfp_unit = exec_unit #SFU
         yield %sfp_unit
      } -> exec_unit

      // L1 Load/Store units.
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      // FIFO datapaths.
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
  }

  // Substitutes exp for an opaque over registers, the way a device hands an
  // intrinsic to a template that implements it. Four registers: a constant the
  // template reads, one for the data each way, and one it computes in. Each is
  // allocated as a single element of the compute type -- how many lanes one
  // holds is the scheduler's to fill in.
  patterns ["pre_scheduling"] {
    pdl.pattern : benefit(1) {
      %f16 = type : f16
      %in = operand : %f16
      %exp = operation "spyreop.exp" (%in : !pdl.value) -> (%f16 : !pdl.type)

      rewrite {
        %reg_file = attribute = "SFU_REG"
        %c0_value = attribute = 1.0 : f16
        %c0_op = operation "arith.constant"
          {"value" = %c0_value, "ktdf_arch.maps_to" = %reg_file}
          -> (%f16 : !pdl.type)
        %c0 = result 0 of %c0_op

        %reg = type : memref<f16, "SFU_REG">
        %no_dyn_operands = attribute = array<i32: 0, 0>

        %in_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %in_reg = result 0 of %in_op
        %in_store = operation "memref.store"
          (%in, %in_reg : !pdl.value, !pdl.value)

        %out_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %out_reg = result 0 of %out_op
        %t0_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %t0 = result 0 of %t0_op

        %template = attribute = "fake_exp"
        %register_names = attribute = ["c0", "in0", "out0", "t0_0"]
        %segments = attribute = array<i32: 2, 2>
        %opaque_op = operation "ktdf.opaque"
          (%c0, %in_reg, %out_reg, %t0
            : !pdl.value, !pdl.value, !pdl.value, !pdl.value)
          {"template_name" = %template, "func_name" = %template,
           "dataflow_scheduler.register_names" = %register_names,
           "operandSegmentSizes" = %segments}

        %unwrap_op = operation "memref.load" (%out_reg : !pdl.value)
          -> (%f16 : !pdl.type)
        %unwrapped = result 0 of %unwrap_op
        replace %exp with (%unwrapped : !pdl.value)
      }
    }
  }

  patterns ["post_scheduling"] {
    // SUM  →  simdreduction_sum
    pdl.pattern : benefit(1) {
      %in  = operand
      %acc = operand
      %generic = operation "linalg.generic" (%in, %acc : !pdl.value, !pdl.value)
      apply_native_constraint "ktdf.is_inner_dim_reduction"(%generic : !pdl.operation)
      %expected_kind = attribute = "sum"
      apply_native_constraint "ktdf.is_reduction_kind"(%generic, %expected_kind : !pdl.operation, !pdl.attribute)

      rewrite {
        %acc_src = apply_native_rewrite "ktdf.subview_source"(%acc : !pdl.value) : !pdl.value
        %new_op = operation "test.simd_reduction_sum"
          (%in, %acc_src : !pdl.value, !pdl.value)
        replace %generic with %new_op
      }
    }

    // ABSMAX  →  test.simd_absmax
    pdl.pattern : benefit(1) {
      %in  = operand
      %acc = operand
      %generic = operation "linalg.generic" (%in, %acc : !pdl.value, !pdl.value)
      apply_native_constraint "ktdf.is_inner_dim_reduction"(%generic : !pdl.operation)
      %expected_kind = attribute = "absmax"
      apply_native_constraint "ktdf.is_reduction_kind"(%generic, %expected_kind : !pdl.operation, !pdl.attribute)

      rewrite {
        %acc_src = apply_native_rewrite "ktdf.subview_source"(%acc : !pdl.value) : !pdl.value
        %new_op = operation "test.simd_reduction_absmax"
          (%in, %acc_src : !pdl.value, !pdl.value)
        replace %generic with %new_op
      }
    }
  }
}
