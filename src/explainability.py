import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_gradcam(model, img_tensor, target_layer):
    """Generates a heatmap showing where the AI is looking."""
    # Placeholder for gradients
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)

    # Forward pass
    features = None
    x = img_tensor.unsqueeze(0)
    for name, module in model.named_children():
        if name == 'fc': break
        x = module(x)
        if name == 'layer4': # ResNet's last conv layer
            features = x
            features.register_hook(save_gradient)
    
    output = model.fc(torch.flatten(x, 1))
    class_idx = torch.argmax(output)
    
    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()
    
    # Weight features by gradients
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap /= np.max(heatmap)
    return heatmap

print("Explainability script ready. Use this to generate heatmaps for results/ folder.")
