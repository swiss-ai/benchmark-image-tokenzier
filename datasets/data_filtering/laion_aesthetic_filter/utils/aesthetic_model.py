import os
import torch
import torch.nn as nn
from os.path import expanduser
from urllib.request import urlretrieve


def get_aesthetic_model(clip_model="vit_l_14", device="cuda"):
    """
    Load the aesthetic predictor model from LAION.
    
    Args:
        clip_model: Either "vit_l_14" or "vit_b_32"
        device: Device to load the model on
    
    Returns:
        Loaded aesthetic model
    """
    home = expanduser("~")
    cache_folder = os.path.join(home, ".cache", "aesthetic_predictor")
    os.makedirs(cache_folder, exist_ok=True)
    
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    
    if not os.path.exists(path_to_model):
        print(f"Downloading aesthetic model weights for {clip_model}...")
        url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        urlretrieve(url_model, path_to_model)
        print(f"Model weights saved to {path_to_model}")
    
    # Create the model architecture
    if clip_model == "vit_l_14":
        model = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        model = nn.Linear(512, 1)
    elif clip_model == "vit_b_16":
        model = nn.Linear(512, 1)
    else:
        raise ValueError(f"Unsupported clip model: {clip_model}")
    
    # Load the weights
    state_dict = torch.load(path_to_model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model