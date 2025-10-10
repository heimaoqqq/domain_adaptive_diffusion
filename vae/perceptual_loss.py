"""
Perceptual Loss implementation for VAE training
Based on LPIPS (Learned Perceptual Image Patch Similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple

# Try to import new weights API (torchvision >= 0.13)
# This handles the deprecation warning for 'pretrained' parameter
try:
    from torchvision.models import VGG16_Weights, AlexNet_Weights
    NEW_WEIGHTS_API = True
except ImportError:
    # Fall back to old API for older torchvision versions
    NEW_WEIGHTS_API = False


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    
    def __init__(self, 
                 feature_layers: List[str] = None,
                 use_normalization: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        if feature_layers is None:
            # Default layers for perceptual loss
            self.feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        else:
            self.feature_layers = feature_layers
            
        # Load pretrained VGG16
        if NEW_WEIGHTS_API:
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device).eval()
        else:
            vgg = models.vgg16(pretrained=True).to(device).eval()
        
        # Extract layers
        self.slices = nn.ModuleList()
        self.layer_names = []
        
        # VGG layers mapping
        vgg_layers = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }
        
        # Create slices
        prev_idx = 0
        for layer_name in self.feature_layers:
            if layer_name in vgg_layers:
                idx = vgg_layers[layer_name]
                self.slices.append(nn.Sequential(*list(vgg.features.children())[prev_idx:idx+1]))
                self.layer_names.append(layer_name)
                prev_idx = idx + 1
                
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization
        self.use_normalization = use_normalization
        if use_normalization:
            # ImageNet normalization
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        # Layer weights (can be tuned)
        self.register_buffer('layer_weights', torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        # Move entire module to device
        self.to(device)
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics"""
        if self.use_normalization:
            return (x - self.mean) / self.std
        return x
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between input and target
        
        Args:
            input: Generated/reconstructed images [B, C, H, W]
            target: Original images [B, C, H, W]
            
        Returns:
            Perceptual loss value
        """
        # Normalize inputs
        input = self.normalize(input)
        target = self.normalize(target)
        
        # Extract features
        loss = 0.0
        x, y = input, target
        
        for i, slice_model in enumerate(self.slices):
            x = slice_model(x)
            y = slice_model(y)
            
            # Compute loss for this layer
            layer_loss = F.mse_loss(x, y)
            loss += self.layer_weights[i] * layer_loss
            
        return loss / len(self.slices)


class LPIPSLoss(nn.Module):
    """
    Simplified LPIPS loss without requiring external lpips library
    Uses pretrained AlexNet features
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        
        # Load pretrained AlexNet
        if NEW_WEIGHTS_API:
            alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
        else:
            alexnet = models.alexnet(pretrained=True).to(device).eval()
        
        # Extract convolutional layers
        self.features = nn.ModuleList([
            alexnet.features[:2],    # Conv1
            alexnet.features[2:5],   # Conv2
            alexnet.features[5:8],   # Conv3
            alexnet.features[8:10],  # Conv4
            alexnet.features[10:12]  # Conv5
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Layer weights (from LPIPS paper)
        self.register_buffer('layer_weights', 
                           torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Move entire module to device
        self.to(device)
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics"""
        return (x - self.mean) / self.std
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss"""
        # Normalize
        input = self.normalize(input)
        target = self.normalize(target)
        
        # Extract and compare features
        loss = 0.0
        x, y = input, target
        
        for i, feature_extractor in enumerate(self.features):
            x = feature_extractor(x)
            y = feature_extractor(y)
            
            # Normalize features
            x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)
            y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-10)
            
            # Compute distance
            diff = (x_norm - y_norm).pow(2).mean(dim=(2, 3))
            loss += self.layer_weights[i] * diff.mean()
            
        return loss


class CombinedPerceptualLoss(nn.Module):
    """Combined loss with MSE, perceptual, and optional style loss"""
    
    def __init__(self,
                 mse_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 style_weight: float = 0.0,
                 loss_type: str = 'vgg',  # 'vgg' or 'lpips'
                 device: str = 'cuda'):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
        # Initialize perceptual loss
        if loss_type == 'vgg':
            self.perceptual_loss = VGGPerceptualLoss(device=device)
        elif loss_type == 'lpips':
            self.perceptual_loss = LPIPSLoss(device=device)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Ensure entire module is on correct device
        self.to(device)
            
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss"""
        b, c, h, w = x.shape
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss
        
        Returns:
            Dictionary with individual loss components
        """
        losses = {}
        
        # MSE loss
        mse_loss = F.mse_loss(input, target)
        losses['mse'] = mse_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(input, target)
        losses['perceptual'] = perceptual_loss
        
        # Style loss (optional)
        if self.style_weight > 0:
            # Use VGG features for style loss
            if hasattr(self.perceptual_loss, 'slices'):
                style_loss = 0.0
                x, y = input, target
                
                for slice_model in self.perceptual_loss.slices:
                    x = slice_model(x)
                    y = slice_model(y)
                    
                    # Compute Gram matrices
                    gram_x = self.gram_matrix(x)
                    gram_y = self.gram_matrix(y)
                    
                    style_loss += F.mse_loss(gram_x, gram_y)
                    
                style_loss /= len(self.perceptual_loss.slices)
                losses['style'] = style_loss
            else:
                losses['style'] = torch.tensor(0.0)
        else:
            losses['style'] = torch.tensor(0.0)
            
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.perceptual_weight * perceptual_loss +
                     self.style_weight * losses['style'])
        
        losses['total'] = total_loss
        
        return losses
