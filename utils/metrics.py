import torch
import torch.nn.functional as F

def test_time_augmentation_inference(model, image, target_size, scales=[0.75, 1.0, 1.25], device='cuda'):
    """
    Perform test-time augmentation by running inference at multiple scales with horizontal flips and aggregating outputs.
    
    Args:
        model: The segmentation model
        image: Input image tensor [C, H, W] (already normalized)
        target_size: Target size (H, W) to resize all outputs to
        scales: List of scale factors to apply
        device: Device to run inference on
    
    Returns:
        Aggregated segmentation logits [1, num_classes, H, W]
    """
    original_size = image.shape[-2:]  # (H, W)
    
    all_outputs = []
    
    with torch.no_grad():
        for scale in scales:
            # Calculate new size
            new_h = int(original_size[0] * scale)
            new_w = int(original_size[1] * scale)
            
            # Resize image
            scaled_image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Run inference on original orientation
            output = model(scaled_image)
            segmentation = output['seg']  # [1, num_classes, H_scaled, W_scaled]
            
            # Resize output back to target size
            segmentation_resized = F.interpolate(
                segmentation,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            all_outputs.append(segmentation_resized)
            
            # Run inference on horizontally flipped image
            scaled_image_flipped = torch.flip(scaled_image, dims=[-1])  # Flip along width dimension
            output_flipped = model(scaled_image_flipped)
            segmentation_flipped = output_flipped['seg']  # [1, num_classes, H_scaled, W_scaled]
            
            # Flip the output back to original orientation
            segmentation_flipped = torch.flip(segmentation_flipped, dims=[-1])
            
            # Resize flipped output back to target size
            segmentation_flipped_resized = F.interpolate(
                segmentation_flipped,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            all_outputs.append(segmentation_flipped_resized)
    
    # Average all outputs
    aggregated_output = torch.stack(all_outputs).mean(dim=0)
    
    return aggregated_output

def update_miou(predictions, targets, intersection_counts, union_counts, num_classes, ignore_index=255):
    predictions = torch.nn.functional.interpolate(
        predictions, size=targets.shape[-2:], mode='bilinear', align_corners=False
    )
    predictions = torch.argmax(predictions, dim=1) # (B, H, W)
    
    # create mask for valid pixels (not ignore_index)
    valid_mask = (targets != ignore_index)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
             
        pred_mask = (predictions == cls) & valid_mask
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum().item()
        union = (pred_mask | target_mask).float().sum().item()
        
        intersection_counts[cls] += intersection
        union_counts[cls] += union