import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

def generate_sam_contours_batch(fastsam_model, images, device):
    """
    Generate FastSAM contours for a batch of images.
    
    Args:
        fastsam_model: FastSAM model instance
        images: List of PIL Images
        device: torch device
    
    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list = []
    contours_y_list = []
    
    for image in images:
        # Resize to 4x downsampled
        image = image.resize((image.size[1] // 4, image.size[0] // 4), Image.Resampling.BILINEAR)
        
        # Save image temporarily for FastSAM (it expects a file path or numpy array)
        image_np = np.array(image)
        
        # Generate masks using FastSAM everything mode
        everything_results = fastsam_model(
            image_np, 
            device=device, 
            retina_masks=True, 
            imgsz=1024,
            conf=0.1,
            iou=0.7,
            verbose=False  # Disable logging
        )
        
        H, W = image_np.shape[:2]
        contours_x = np.zeros((H, W - 1), dtype=bool)
        contours_y = np.zeros((H - 1, W), dtype=bool)
        
        # Extract segmentation masks from FastSAM results
        # FastSAM results are YOLO-style, masks are in results[0].masks.data
        if everything_results[0].masks is not None:
            masks = everything_results[0].masks.data.cpu().numpy()  # [num_masks, H, W]
            for mask in masks:
                # Convert to boolean and ensure correct shape
                mask_bool = mask.astype(bool)
                if mask_bool.shape[0] == H and mask_bool.shape[1] == W:
                    contours_x |= np.logical_xor(mask_bool[:, :-1], mask_bool[:, 1:])  # shape: (H, W-1)
                    contours_y |= np.logical_xor(mask_bool[:-1, :], mask_bool[1:, :])  # shape: (H-1, W)
        
        # Convert to tensors
        contours_x_tensor = torch.from_numpy(contours_x).float()  # [H, W-1]
        contours_y_tensor = torch.from_numpy(contours_y).float()  # [H-1, W]
        contours_x_list.append(contours_x_tensor)
        contours_y_list.append(contours_y_tensor)
    
    # Stack into batch tensors
    contours_x_batch = torch.stack(contours_x_list)  # [B, H, W-1]
    contours_y_batch = torch.stack(contours_y_list)  # [B, H-1, W]
    
    return contours_x_batch, contours_y_batch