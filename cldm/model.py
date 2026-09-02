import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def resolve_state_dict_path(path):
    if os.path.exists(path):
        return path

    root, extension = os.path.splitext(path)
    alternatives = []
    if extension.lower() == ".ckpt":
        alternatives.append(root + ".safetensors")
    elif extension.lower() == ".safetensors":
        alternatives.append(root + ".ckpt")

    for alternative in alternatives:
        if os.path.exists(alternative):
            return alternative

    external_model_dir = os.environ.get("GAUSSIANOBJECT_MODEL_DIR")
    if external_model_dir:
        external_path = os.path.join(external_model_dir, os.path.basename(path))
        if os.path.exists(external_path):
            return external_path

        external_root, external_extension = os.path.splitext(external_path)
        external_alternatives = []
        if external_extension.lower() == ".ckpt":
            external_alternatives.append(external_root + ".safetensors")
        elif external_extension.lower() == ".safetensors":
            external_alternatives.append(external_root + ".ckpt")

        for alternative in external_alternatives:
            if os.path.exists(alternative):
                return alternative

    return path


def load_state_dict(ckpt_path, location='cpu'):
    ckpt_path = resolve_state_dict_path(ckpt_path)
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
