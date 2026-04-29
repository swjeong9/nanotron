from dataclasses import dataclass
from typing import List, Optional

from nanotron.config.utils_config import (
    cast_str_to_pipeline_engine,
)
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


@dataclass
class ParallelismArgs:
    """Arguments related to TP/PP/DP

    Args:
        dp: Number of DP replicas
        pp: Number of PP stages
        tp: Number of TP replicas
        expert_parallel_size: Number of expert parallel replicas (used only for MoEs)
        pp_engine: Pipeline engine to use between "1f1b" and "afab"
        tp_mode: TP mode to use between "all_reduce" and "reduce_scatter": all_reduce is normal, reduce_scatter activate sequence parallelism
        tp_linear_async_communication: Whether to use async communication in TP linear layers
        recompute_layer: Whether to recompute each Transformer layer to save memory.
        pp_layer_partition: Optional list of decoder-layer counts per PP stage, e.g. [4, 8, 6]
            for PP=3 with 18 decoder layers. When set, overrides the default
            compute-cost-balanced split: only decoder layers are partitioned by the list,
            embedding goes to stage 0 and final_layer_norm/lm_head/loss go to the last stage.
            Must satisfy len == pp and sum == num_hidden_layers (the latter is checked once
            the model config is available, in the trainer). Default None preserves the
            existing cost-based behavior.
    """

    dp: int
    pp: int
    tp: int
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    tp_linear_async_communication: Optional[bool] = None
    recompute_layer: bool = False
    tp_recompute_allgather: bool = True

    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    pp_layer_partition: Optional[List[int]] = None

    def __post_init__(self):
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]

        if self.pp_layer_partition is not None:
            assert len(self.pp_layer_partition) == self.pp, (
                f"pp_layer_partition has length {len(self.pp_layer_partition)} but pp={self.pp}"
            )
            assert all(n >= 1 for n in self.pp_layer_partition), (
                f"pp_layer_partition entries must be >= 1, got {self.pp_layer_partition}"
            )
