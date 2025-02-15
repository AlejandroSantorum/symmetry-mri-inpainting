from typing import Dict


def get_train_defaults() -> Dict:
    """
    Default arguments for training the model.

    Returns:
        dict: A dictionary containing default values for training parameters.
            - schedule_sampler (str): The type of schedule sampler to use (default: "uniform").
            - lr (float): The learning rate for the optimizer (default: 1e-4).
            - weight_decay (float): The weight decay (L2 regularization) coefficient (default: 0.0).
            - lr_anneal_steps (int): The number of steps to anneal the learning rate over (default: 0, no annealing).
            - batch_size (int): The number of samples per batch (default: 1).
            - microbatch (int): The size of microbatches to use (default: -1, disables microbatches).
            - ema_rate (str): The exponential moving average (EMA) rate for model parameters (default: "0.9999", comma-separated list of EMA values).
            - log_interval (int): The interval (in steps) at which to log training progress (default: 1000).
            - save_interval (int): The interval (in steps) at which to save model checkpoints (default: 10000).
            - resume_checkpoint (str): Path to a checkpoint to resume training from (default: "", no checkpoint).
            - use_fp16 (bool): Whether to use 16-bit floating point precision (default: False).
            - fp16_scale_growth (float): The growth factor for dynamic loss scaling in FP16 training (default: 1e-3).
    """
    return dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )


def get_model_and_diffusion_defaults() -> Dict:
    """
    Get the default arguments for model and diffusion training.
    """
    model_defaults = get_model_defaults()
    diffusion_defaults = get_diffusion_defaults()
    return {**model_defaults, **diffusion_defaults}


def get_model_defaults() -> Dict:
    """
    Default arguments for U-Net model training.

    Returns:
        dict: A dictionary containing default values for model training parameters.
            - image_size (int): The size of the input images (default: 64).
            - num_model_channels (int): The number of features for the time embedding (default: 128).
            - num_in_channels (int): The number of input channels (default: 3, e.g., RGB images).
            - num_out_channels (int): The number of output channels (default: 2).
            - num_res_blocks (int): The number of residual blocks in the model (default: 2).
            - num_heads (int): The number of attention heads (default: 4).
            - num_heads_upsample (int): The number of attention heads for upsampling (default: -1, use num_heads).
            - num_head_channels (int): The number of channels per attention head (default: -1, auto-compute).
            - attention_resolutions (str): The resolutions at which to apply attention (default: "16,8").
            - channel_mult (str): Channel multiplier for each resolution (default: "", use default multipliers).
            - dropout (float): Dropout rate (default: 0.0).
            - class_cond (bool): Whether to use class conditioning (default: False).
            - use_checkpoint (bool): Whether to use gradient checkpointing (default: False).
            - use_scale_shift_norm (bool): Whether to use scale-shift normalization (default: True).
            - resblock_updown (bool): Whether to use residual blocks for up/downsampling (default: False).
            - use_fp16 (bool): Whether to use 16-bit floating point precision (default: False).
            - use_new_attention_order (bool): Whether to use a new attention order (default: False).
    """
    return dict(
        image_size=64,
        num_model_channels=128,
        num_in_channels=3,
        num_out_channels=2,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )


def get_diffusion_defaults() -> Dict:
    """
    Default arguments for denosing diffusion training.

    Returns:
        dict: A dictionary containing default values for diffusion training parameters.
            - learn_sigma (bool): Whether to learn the noise standard deviation (default: False).
            - diffusion_steps (int): The number of diffusion steps (default: 1000).
            - noise_schedule (str): The noise schedule type (default: "linear").
            - timestep_respacing (str): The respacing of timesteps (default: "", use default spacing).
            - use_kl (bool): Whether to use KL divergence loss (default: False).
            - predict_xstart (bool): Whether to predict the starting image (default: False).
            - rescale_timesteps (bool): Whether to rescale timesteps (default: False).
            - rescale_learned_sigmas (bool): Whether to rescale learned sigmas (default: False).
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
