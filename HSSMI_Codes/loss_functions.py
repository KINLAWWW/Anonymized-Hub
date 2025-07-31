import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gradcam import GradCAM

def get_prefrontal_mask(grid_size=(3, 3)):
    mask = np.zeros(grid_size, dtype=np.float32)
    mask[1,0] = 1 
    mask[1,1] = 1  
    mask[1,2] = 1
    return torch.tensor(mask, dtype=torch.float32)

def reshape_transform(tensor, time=16, height=3, width=3):
    return tensor


def compute_attention_map(model, input_tensor, layer_idx=-1, block_idx=-1):
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    target_layers = [model.patch_embed.proj]
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = torch.tensor(grayscale_cam, device=input_tensor.device)
    grayscale_cam = F.relu(grayscale_cam)
    grayscale_cam /= torch.amax(grayscale_cam, dim=(1, 2), keepdim=True) + 1e-8
    
    return grayscale_cam

def region_wise_loss(outputs, labels, activation_maps, mask, device):
    ce_loss = F.cross_entropy(outputs, labels, reduction='mean')
    
    mask = mask.to(device)
    batch_size = activation_maps.shape[0]
    P = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        H_i = activation_maps[i] 
        masked_importance = (H_i * mask).sum()
        P[i] = torch.sigmoid(masked_importance)
    
    region_loss = ce_loss - P.mean()
    return region_loss

def sample_wise_loss(outputs, labels, activation_maps, mask, device):
    batch_size = outputs.shape[0]
    mask = mask.to(device)
    W = torch.zeros(batch_size, device=device)
    Z = mask.sum()
    
    for i in range(batch_size):
        H_i = activation_maps[i]
        masked_importance = (H_i * mask).sum()
        W[i] = torch.sigmoid(-masked_importance / (Z + 1e-8))
    
    log_probs = F.log_softmax(outputs, dim=1)
    losses = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        losses[i] = -log_probs[i, labels[i]] * W[i]
    
    sample_loss = losses.mean()
    return sample_loss

def combined_loss(outputs, labels, activation_maps, mask, device, region_weight=0.8):
    region_loss = region_wise_loss(outputs, labels, activation_maps, mask, device)
    sample_loss = sample_wise_loss(outputs, labels, activation_maps, mask, device)
    loss = region_weight * region_loss + (1 - region_weight) * sample_loss

    return loss