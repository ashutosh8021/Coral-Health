import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def __call__(self, input_tensor, targets=None):
        # Register hooks
        handle_backward = self.target_layers[0].register_backward_hook(self.save_gradient)
        handle_forward = self.target_layers[0].register_forward_hook(self.save_activation)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get the target class
        if targets is None:
            targets = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, targets[0]]
        class_loss.backward()
        
        # Calculate GradCAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam = cam / torch.max(cam)
        
        # Remove hooks
        handle_backward.remove()
        handle_forward.remove()
        
        return cam.detach().cpu().numpy()[None, :]

def generate_gradcam(model, input_tensor):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    return grayscale_cam

def show_cam_on_image(img, mask, use_rgb=False, colormap=cv2.COLORMAP_JET):
    """Show GradCAM heatmap on image"""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")
    
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
