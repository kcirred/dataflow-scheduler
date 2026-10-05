add_dataflow_scheduler_library(DataflowSchedulerKTDFAnalysis
    ApplicableUnits.cpp
    GlobalStageDAG.cpp
    LoopTiling.cpp
    PipelineScope.cpp
    ReductionChunkAnalysis.cpp
    Reuse.cpp
    StageGrouping.cpp
    TileSizeConstraints.cpp
    Utils.cpp
  
  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF/Analysis

  GROUP DIALECT
  EXPORT_NAME KTDFAnalysis

  LINK_LIBS PUBLIC
    DataflowScheduler::Analysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRSCFUtils
)

mlir_target_link_libraries(DataflowSchedulerKTDFAnalysis PUBLIC
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFUtils
)
