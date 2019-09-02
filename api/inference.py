import torch
from api.commons import get_model
import os

def get_painting_tensor(photo, style):
    """Forward function used in test time.

    This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
    It also calls <compute_visuals> to produce additional visualization results
    """
    model = get_model(style)
    with torch.no_grad():
        return model.forward(photo)
   