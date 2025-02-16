import argparse
from typing import Dict, Union

from symmetry_mri_inpainting.model.defaults import (
    get_model_and_diffusion_defaults,
    get_train_defaults,
)


def get_create_eval_argparser(
    parser: argparse.ArgumentParser = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Evaluate a symmetry-aware denoising diffusion model on MRI images for inpainting."
        )
    add_dict_to_argparser(parser, get_model_and_diffusion_defaults())
    return parser


def get_create_train_argparser(
    parser: argparse.ArgumentParser = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Train a symmetry-aware denoising diffusion model on MRI images for inpainting."
        )
    add_dict_to_argparser(parser, get_train_defaults())
    add_dict_to_argparser(parser, get_model_and_diffusion_defaults())
    return parser


def add_dict_to_argparser(parser: argparse.ArgumentParser, default_dict: Dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(value: Union[str, bool]) -> bool:
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"A boolean value expected, but '{value}' was given."
        )
