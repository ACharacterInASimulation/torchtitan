from torchtitan.experiments.deepseek_engram.model import DeepSeekEngramModel
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3


def parallelize_deepseek_engram(
    model: DeepSeekEngramModel,
    *,
    parallel_dims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    if parallel_dims.tp_enabled and parallelism.enable_sequence_parallel:
        raise NotImplementedError(
            "DeepSeek Engram v1 requires sequence parallel to be disabled when TP is enabled."
        )

    return parallelize_deepseekv3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
