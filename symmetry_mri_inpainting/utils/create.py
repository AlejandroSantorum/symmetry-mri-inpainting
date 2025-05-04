import copy
from typing import Dict

from symmetry_mri_inpainting.model import gaussian_diffusion as gd
from symmetry_mri_inpainting.model.defaults import (
    get_diffusion_defaults,
    get_model_defaults,
)
from symmetry_mri_inpainting.model.resample import UniformSampler
from symmetry_mri_inpainting.model.respace import SpacedDiffusion, space_timesteps
from symmetry_mri_inpainting.model.unet import UNetModel


def create_named_schedule_sampler(name, diffusion, maxt):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion, maxt)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


def create_unet_model(args: Dict) -> UNetModel:
    """
    Create a U-Net model for inpainting.
    """
    model_defaults = get_model_defaults()

    model_args = copy.deepcopy(model_defaults)
    for arg in model_defaults:
        if arg in args:
            model_args[arg] = args[arg]

    return _create_unet_model(**model_args)


def create_gaussian_diffusion(args: Dict) -> SpacedDiffusion:
    """
    Create a Gaussian diffusion model for inpainting.
    """
    diffusion_defaults = get_diffusion_defaults()

    diffusion_args = copy.deepcopy(diffusion_defaults)
    for arg in diffusion_defaults:
        if arg in args:
            diffusion_args[arg] = args[arg]

    return _create_gaussian_diffusion(**diffusion_args)


def _create_unet_model(
    image_size,
    num_model_channels,
    num_in_channels,
    num_out_channels,
    num_res_blocks,
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        model_channels=num_model_channels,
        in_channels=num_in_channels,
        out_channels=num_out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(2 if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def _create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
