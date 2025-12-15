import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import RAE
import torch.nn as nn
from omegaconf import OmegaConf
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "emas" in state_dict:
            ema_decays = list(state_dict["emas"].keys())
            print(f"Found {len(ema_decays)} EMA decays: {ema_decays}")
            state_dict = state_dict["emas"][ema_decays[0]]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
            state_dict = state_dict["model"]
        
        # Remove _orig_mod prefix from state_dict keys. This is a known issue with torch.compile
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[len("_orig_mod."):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model

