"""
Test script to evaluate model performance on boundary misalignment.

This script:
1. Creates a synthetic square image with 2 colors (red square on green background)
2. Downsamples the image by 4x (512x512 -> 128x128)
3. Generates ground truth contours from the downsampled mask
4. Adds noise to boundaries by shifting a percentage of contour pixels
5. Trains the model using:
   - Ground truth labels (directly from mask)
   - Noisy boundaries (shifted contours)
6. Visualizes results at both 4x downsampled and original resolution
   to see the effect of misaligned boundaries on model convergence

Hyperparameters (configurable in main()):
- NOISE_PERCENTAGE: Percentage of boundaries to shift (0.0-1.0)
- NOISE_SHIFT: Number of pixels to shift boundaries (1 or 2)
- NUM_EPOCHS: Number of training epochs
- LEARNING_RATE: Learning rate for training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import yaml

# Add paths
DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
sys.path.append(DINOV3_LOCATION)

from model.dino import DinoWSSS
from utils.loss import CollisionCrossEntropyLoss, PottsLoss
from torchvision import transforms


def create_synthetic_square_image(size=(512, 512), square_size=256, square_pos=(128, 128), 
                                  color1=(255, 0, 0), color2=(0, 255, 0)):
    """
    Create a synthetic image with a square of one color on a background of another color.
    
    Args:
        size: (H, W) tuple for image size
        square_size: Size of the square
        square_pos: (y, x) position of top-left corner of square
        color1: RGB color for square (0-255)
        color2: RGB color for background (0-255)
    
    Returns:
        image: PIL Image
        mask: numpy array [H, W] with class indices (0=background, 1=square)
    """
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    
    y_start, x_start = square_pos
    y_end = min(y_start + square_size, H)
    x_end = min(x_start + square_size, W)
    
    # Fill square with color1 (class 1)
    image[y_start:y_end, x_start:x_end] = color1
    mask[y_start:y_end, x_start:x_end] = 1
    
    # Fill background with color2 (class 0)
    image[mask == 0] = color2
    
    return Image.fromarray(image), mask


def downsample_image_and_mask(image, mask, factor=4):
    """
    Downsample image and mask by a factor.
    
    Args:
        image: PIL Image
        mask: numpy array [H, W]
        factor: Downsampling factor
    
    Returns:
        image_down: PIL Image (downsampled)
        mask_down: numpy array (downsampled, nearest neighbor)
    """
    H, W = image.size[1], image.size[0]
    H_down = H // factor
    W_down = W // factor
    
    # Downsample image with bilinear
    image_down = image.resize((W_down, H_down), Image.Resampling.BILINEAR)
    
    # Downsample mask with nearest neighbor
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    mask_down_tensor = F.interpolate(mask_tensor, size=(H_down, W_down), mode='nearest')
    mask_down = mask_down_tensor.squeeze().numpy().astype(np.uint8)
    
    return image_down, mask_down


def generate_jittered_contours(mask, noise_percentage=0.1, noise_shift=1, device=None):
    """
    Generate jittered contours by adding pixels around the mask boundary within the shift range.
    These contours will have 0 pairwise loss for the modified mask, but won't match the original boundaries.
    
    The jitter is created by:
    1. Finding all pixels within noise_shift distance from the mask boundary (currently 0)
    2. Randomly selecting noise_percentage of those pixels
    3. Setting them to 1 to expand the mask
    4. Generating contours from the modified mask
    
    Args:
        mask: numpy array [H, W] with class indices (0=background, 1=square)
        noise_percentage: Percentage of boundary pixels to add (0-1). Higher = more jitter.
        noise_shift: Maximum distance from boundary to consider (1 or 2 pixels)
        device: torch device (optional, for consistency with other functions)
    
    Returns:
        jittered_contours_x: numpy array [H, W-1]
        jittered_contours_y: numpy array [H-1, W]
        jittered_mask: numpy array [H, W] - the modified mask used to generate contours
    """
    # Create a copy of the mask
    jittered_mask = mask.copy().astype(np.uint8)
    H, W = jittered_mask.shape
    
    # Find all pixels that are within noise_shift distance from the mask boundary
    # These are pixels that are currently 0 (background) but adjacent to pixels that are 1 (square)
    
    # Create a list of candidate pixels (background pixels near the boundary)
    candidate_pixels = []
    
    # Check all pixels in the image
    for h in range(H):
        for w in range(W):
            # Only consider background pixels (value 0)
            if jittered_mask[h, w] == 0:
                # Check if this pixel is within noise_shift distance from a square pixel
                is_near_boundary = False
                
                # Check neighbors within noise_shift distance
                for dh in range(-noise_shift, noise_shift + 1):
                    for dw in range(-noise_shift, noise_shift + 1):
                        # Skip the pixel itself
                        if dh == 0 and dw == 0:
                            continue
                        
                        # Check Manhattan distance (L1) is within noise_shift
                        if abs(dh) + abs(dw) <= noise_shift:
                            nh, nw = h + dh, w + dw
                            # Check if neighbor is within bounds and is part of the square
                            if 0 <= nh < H and 0 <= nw < W and jittered_mask[nh, nw] == 1:
                                is_near_boundary = True
                                break
                    
                    if is_near_boundary:
                        break
                
                if is_near_boundary:
                    candidate_pixels.append((h, w))
    
    # Randomly select noise_percentage of candidate pixels to add to the mask
    if len(candidate_pixels) > 0:
        if noise_percentage > 0.0:
            num_to_add = max(1, int(len(candidate_pixels) * noise_percentage))
            selected_pixels = np.random.choice(len(candidate_pixels), size=num_to_add, replace=False)
            
            # Set selected pixels to 1
            for idx in selected_pixels:
                h, w = candidate_pixels[idx]
                jittered_mask[h, w] = 1
    
    # Generate contours from the jittered mask
    # These contours will be consistent with the jittered_mask (0 pairwise loss for that mask)
    # because they are the exact boundaries of that mask
    jittered_contours_x = (jittered_mask[:, :-1] != jittered_mask[:, 1:]).astype(np.float32)  # [H, W-1]
    jittered_contours_y = (jittered_mask[:-1, :] != jittered_mask[1:, :]).astype(np.float32)  # [H-1, W]
    
    return jittered_contours_x, jittered_contours_y, jittered_mask


def get_ground_truth_contours(mask, device):
    """
    Get ground truth contours from mask without any downsampling.
    
    Args:
        mask: numpy array [H, W] with class indices
        device: torch device
    
    Returns:
        contours_x: torch.Tensor [H, W-1]
        contours_y: torch.Tensor [H-1, W]
    """
    # Detect horizontal edges (contours_x): compare each pixel with its right neighbor
    contours_x = (mask[:, :-1] != mask[:, 1:]).astype(np.float32)  # [H, W-1]
    
    # Detect vertical edges (contours_y): compare each pixel with its bottom neighbor
    contours_y = (mask[:-1, :] != mask[1:, :]).astype(np.float32)  # [H-1, W]
    
    contours_x_tensor = torch.from_numpy(contours_x).to(device)  # [H, W-1]
    contours_y_tensor = torch.from_numpy(contours_y).to(device)  # [H-1, W]
    
    return contours_x_tensor, contours_y_tensor


def create_synthetic_dataset(image, mask, transform):
    """
    Create a dataset-like object for a single image.
    
    Args:
        image: PIL Image
        mask: numpy array [H, W]
        transform: torchvision transform
    
    Returns:
        transformed_image: torch.Tensor [C, H, W]
        target: torch.Tensor [H, W]
    """
    # Transform image
    transformed_image = transform(image)
    
    # Convert mask to tensor
    target = torch.from_numpy(mask).long()
    
    return transformed_image, target


def visualize_results(image_orig, image_down, mask_gt, predictions_down, predictions_orig,
                     contours_gt_x, contours_gt_y, contours_noisy_x, contours_noisy_y,
                     jittered_mask, jittered_mask_soft_prob, epoch, output_dir, noise_percentage, noise_shift, soft_prob):
    """
    Visualize training results at different resolutions.
    
    Args:
        image_orig: PIL Image (original resolution)
        image_down: PIL Image (downsampled)
        mask_gt: numpy array [H_down, W_down] ground truth mask
        predictions_down: torch.Tensor [C, H_down, W_down] model predictions at downsampled resolution
        predictions_orig: torch.Tensor [C, H_orig, W_orig] model predictions at original resolution
        contours_gt_x, contours_gt_y: Ground truth contours
        contours_noisy_x, contours_noisy_y: Noisy contours used for training
        jittered_mask: numpy array [H_down, W_down] jittered mask used to generate noisy contours
        jittered_mask_soft_prob: numpy array [H_down, W_down] soft probability mask (foreground class probability)
        epoch: Current epoch
        output_dir: Directory to save visualizations
        noise_percentage: Percentage of noisy boundaries
        noise_shift: Shift amount in pixels
        soft_prob: Soft probability value for foreground class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert predictions to numpy and get soft probabilities
    if isinstance(predictions_down, torch.Tensor):
        pred_down_np = torch.softmax(predictions_down, dim=0).cpu().numpy()  # [C, H, W]
        pred_down_prob_square = pred_down_np[1]  # Probability of square class [H, W]
    
    if isinstance(predictions_orig, torch.Tensor):
        pred_orig_np = torch.softmax(predictions_orig, dim=0).cpu().numpy()  # [C, H, W]
        pred_orig_prob_square = pred_orig_np[1]  # Probability of square class [H, W]
    
    # Convert contours to numpy if needed
    if isinstance(contours_gt_x, torch.Tensor):
        contours_gt_x = contours_gt_x.cpu().numpy()
    if isinstance(contours_gt_y, torch.Tensor):
        contours_gt_y = contours_gt_y.cpu().numpy()
    if isinstance(contours_noisy_x, torch.Tensor):
        contours_noisy_x = contours_noisy_x.cpu().numpy()
    if isinstance(contours_noisy_y, torch.Tensor):
        contours_noisy_y = contours_noisy_y.cpu().numpy()
    
    # Create figure with subplots (4 rows x 4 columns)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    # Row 1: Original resolution
    axes[0, 0].imshow(image_orig)
    axes[0, 0].set_title('Original Image')
    
    # Upsample ground truth mask to original resolution
    mask_gt_orig = F.interpolate(
        torch.from_numpy(mask_gt).unsqueeze(0).unsqueeze(0).float(),
        size=(image_orig.size[1], image_orig.size[0]),
        mode='nearest'
    ).squeeze().numpy()
    axes[0, 1].imshow(mask_gt_orig, cmap='gray')
    axes[0, 1].set_title('GT Mask (upsampled)')
    
    # Resize probability to match original image size if needed
    H_orig, W_orig = image_orig.size[1], image_orig.size[0]
    pred_orig_prob_square_resized = pred_orig_prob_square
    
    # Show soft probability as grayscale
    axes[0, 2].imshow(pred_orig_prob_square_resized, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Soft Prediction (Original, Epoch {epoch})')
    
    # Blend soft prediction with image
    overlay_orig = np.array(image_orig).copy().astype(float)
    # Blend red channel with probability (red square class)
    overlay_orig[:, :, 0] = overlay_orig[:, :, 0] * (1 - 0.6 * pred_orig_prob_square_resized) + 255 * 0.6 * pred_orig_prob_square_resized
    axes[0, 3].imshow(overlay_orig.astype(np.uint8))
    axes[0, 3].set_title('Soft Overlay (Original)')
    
    # Row 2: Downsampled resolution
    axes[1, 0].imshow(image_down)
    axes[1, 0].set_title('Downsampled Image (4x)')
    
    axes[1, 1].imshow(mask_gt, cmap='gray')
    axes[1, 1].set_title('GT Mask (Downsampled)')
    
    # Resize probability to match downsampled image size if needed
    H_down_img, W_down_img = image_down.size[1], image_down.size[0]
    pred_down_prob_square_resized = pred_down_prob_square
    
    # Show soft probability as grayscale
    axes[1, 2].imshow(pred_down_prob_square_resized, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Soft Prediction (Downsampled, Epoch {epoch})')
    
    # Blend soft prediction with downsampled image
    overlay_down = np.array(image_down).copy().astype(float)
    # Blend red channel with probability (red square class)
    overlay_down[:, :, 0] = overlay_down[:, :, 0] * (1 - 0.6 * pred_down_prob_square_resized) + 255 * 0.6 * pred_down_prob_square_resized
    axes[1, 3].imshow(overlay_down.astype(np.uint8))
    axes[1, 3].set_title('Soft Overlay (Downsampled)')  
    
    # Row 3: Contours
    axes[2, 0].imshow(contours_gt_x, cmap='hot')
    axes[2, 0].set_title('GT Contours X')
    
    axes[2, 1].imshow(contours_gt_y, cmap='hot')
    axes[2, 1].set_title('GT Contours Y')
    
    axes[2, 2].imshow(contours_noisy_x, cmap='hot')
    axes[2, 2].set_title(f'Noisy Contours X ({noise_percentage*100:.1f}%, shift={noise_shift})')
    
    axes[2, 3].imshow(contours_noisy_y, cmap='hot')
    axes[2, 3].set_title(f'Noisy Contours Y ({noise_percentage*100:.1f}%, shift={noise_shift})')
    
    # Row 4: Mask comparisons
    axes[3, 0].imshow(mask_gt, cmap='gray')
    axes[3, 0].set_title('GT Mask (Downsampled)')
    
    # Show soft probability mask (foreground class probability)
    axes[3, 1].imshow(jittered_mask_soft_prob, cmap='gray', vmin=0, vmax=1)
    axes[3, 1].set_title(f'Jittered Mask (Soft Prob, fg={soft_prob:.1f})')
    
    # Difference between GT and jittered mask (using hard mask)
    mask_diff = np.abs(mask_gt.astype(float) - jittered_mask.astype(float))
    axes[3, 2].imshow(mask_diff, cmap='hot')
    axes[3, 2].set_title('Difference (GT - Jittered)')
    
    # Overlay soft probability mask on downsampled image
    overlay_jittered = np.array(image_down).copy()
    overlay_jittered[:, :, 0] = overlay_jittered[:, :, 0] * (1 - 0.6 * jittered_mask_soft_prob) + 255 * 0.6 * jittered_mask_soft_prob
    axes[3, 3].imshow(overlay_jittered.astype(np.uint8))
    axes[3, 3].set_title('Soft Mask Overlay')
    
    plt.suptitle(f'Boundary Misalignment Test - Epoch {epoch} (Noise: {noise_percentage*100:.1f}%, Shift: {noise_shift}px)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'boundary_test_epoch_{epoch:03d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def main():
    # Configuration
    config_path = 'config.yaml'
    config = yaml.safe_load(open(config_path, 'r'))
    
    # ========== HYPERPARAMETERS ==========
    # Boundary misalignment parameters
    NOISE_PERCENTAGE = 0.5  # Percentage of boundaries to shift (0.0-1.0), e.g., 0.2 = 20%
    NOISE_SHIFT = 1  # Number of pixels to shift boundaries (1 or 2)
    SOFT_PROB = 0.8  # Soft probability for foreground class (background gets 1.0 - SOFT_PROB)
    
    # Training parameters
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 1
    
    # Synthetic image parameters
    IMAGE_SIZE = (512, 512)
    SQUARE_SIZE = 256
    SQUARE_POS = (128, 128)
    # ======================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic image
    print("Creating synthetic square image...")
    image_orig, mask_orig = create_synthetic_square_image(
        size=IMAGE_SIZE,
        square_size=SQUARE_SIZE,
        square_pos=SQUARE_POS,
        color1=(255, 0, 0),  # Red square
        color2=(0, 255, 0)   # Green background
    )
    
    # Downsample by 4x
    print("Downsampling image and mask...")
    image_down, mask_down = downsample_image_and_mask(image_orig, mask_orig, factor=4)
    
    H_down, W_down = mask_down.shape
    print(f"Downsampled size: {H_down}x{W_down}")
    
    # Get ground truth contours
    print("Generating ground truth contours...")
    contours_gt_x, contours_gt_y = get_ground_truth_contours(mask_down, device)
    contours_gt_x_np = contours_gt_x.cpu().numpy()
    contours_gt_y_np = contours_gt_y.cpu().numpy()
    
    # Generate jittered contours by modifying the mask
    print(f"Generating jittered contours ({NOISE_PERCENTAGE*100:.1f}%, shift={NOISE_SHIFT}px)...")
    np.random.seed(42)  # For reproducibility
    contours_noisy_x_np, contours_noisy_y_np, jittered_mask = generate_jittered_contours(
        mask_down,
        noise_percentage=NOISE_PERCENTAGE, 
        noise_shift=NOISE_SHIFT,
        device=device
    )
    contours_noisy_x = torch.from_numpy(contours_noisy_x_np).float().to(device)
    contours_noisy_y = torch.from_numpy(contours_noisy_y_np).float().to(device)
    
    # Convert jittered mask to tensor for use in training
    jittered_mask_tensor = torch.from_numpy(jittered_mask).long().to(device)  # [H_down, W_down]
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset - use original image for model input (model expects original resolution)
    # Model will internally downsample to 4x, so output will be at 128x128
    transformed_image, _ = create_synthetic_dataset(image_orig, mask_orig, transform)
    transformed_image = transformed_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Target should be at the model output resolution (4x downsampled = 128x128)
    # So we use mask_down
    target = torch.from_numpy(mask_down).long().to(device)  # [H_down, W_down]
    
    # Initialize model
    print("Initializing model...")
    model = DinoWSSS(
        backbone_name=config['model']['backbone_name'],
        num_transformer_blocks=config['model']['num_transformer_blocks'],
        num_conv_blocks=config['model']['num_conv_blocks'],
        out_channels=2,  # 2 classes
        use_bottleneck=config['model']['use_bottleneck'],
        use_transpose_conv=config['model']['use_transpose_conv']
    ).to(device)
    model.backbone.eval()
    model.dinotxt_head.eval()
    
    # Initialize optimizer
    optimizer_params = [
        {'params': model.transformer_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.ln.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.lin_classifier.parameters(), 'lr': LEARNING_RATE},
    ]
    if config['model']['use_transpose_conv']:
        optimizer_params.append({'params': model.upsample_conv1.parameters(), 'lr': LEARNING_RATE})
        optimizer_params.append({'params': model.upsample_conv2.parameters(), 'lr': LEARNING_RATE})
    
    optimizer = torch.optim.SGD(
        params=optimizer_params,
        lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001
    )
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    output_dir = 'output_boundary_test'
    os.makedirs(output_dir, exist_ok=True)
    
    losses = []
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        model.backbone.eval()
        model.dinotxt_head.eval()
        
        optimizer.zero_grad()
        
        # Forward pass
        model_outputs = model(transformed_image)
        segmentations = model_outputs['seg']  # [B, C, H, W]
        
        # Use jittered mask for unary potentials (instead of ground truth)
        B, C, H_seg, W_seg = segmentations.shape
        
        # Resize jittered mask to match segmentation size if needed
        jittered_mask_resized = jittered_mask_tensor.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H_down, W_down]
        jittered_mask_resized = jittered_mask_resized.squeeze(0).long()  # [1, H_seg, W_seg]
        
        # Convert to soft probabilities (jittered mask)
        # Use soft probabilities: foreground class gets SOFT_PROB, background gets 1.0 - SOFT_PROB
        # For valid probability distribution: if mask==1 (foreground): [bg=1-SOFT_PROB, fg=SOFT_PROB]
        #                                     if mask==0 (background): [bg=SOFT_PROB, fg=1-SOFT_PROB]
        pseudolabel_probs = torch.zeros((B, 2, H_seg, W_seg), dtype=torch.float32, device=device)
        for b in range(B):
            # Foreground pixels (jittered_mask == 1): [bg=1-SOFT_PROB, fg=SOFT_PROB]
            # Background pixels (jittered_mask == 0): [bg=SOFT_PROB, fg=1-SOFT_PROB]
            is_foreground = (jittered_mask_resized[b] == 1).float()
            pseudolabel_probs[b, 0] = is_foreground * (1.0 - SOFT_PROB) + (1.0 - is_foreground) * SOFT_PROB  # background class
            pseudolabel_probs[b, 1] = is_foreground * SOFT_PROB + (1.0 - is_foreground) * (1.0 - SOFT_PROB)  # foreground class
        
        # Unary loss
        unary_loss = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs)
        
        if epoch < 200:
            pairwise_loss = torch.tensor(0.0, device=device)
        else:
            pairwise_loss = PottsLoss(
                'quadratic',
                segmentations,
                contours_noisy_x.unsqueeze(0),
                contours_noisy_y.unsqueeze(0),
                use_color_diff=False
            )
        
        total_loss = unary_loss + pairwise_loss
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Get predictions at downsampled resolution (model output is already 4x downsampled)
                pred_down = model(transformed_image)['seg'][0]  # [C, H_down, W_down]
                
                # Get predictions at original resolution by upsampling the model output
                pred_orig = F.interpolate(
                    pred_down.unsqueeze(0),
                    size=(image_orig.size[1], image_orig.size[0]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # [C, H_orig, W_orig]
                
                # Create soft probability mask for visualization (foreground class probability)
                jittered_mask_soft_prob = jittered_mask.astype(float) * SOFT_PROB + (1.0 - jittered_mask.astype(float)) * (1.0 - SOFT_PROB)
                
                visualize_results(
                    image_orig, image_down, mask_down,
                    pred_down, pred_orig,
                    contours_gt_x_np, contours_gt_y_np,
                    contours_noisy_x_np, contours_noisy_y_np,
                    jittered_mask, jittered_mask_soft_prob,
                    epoch + 1, output_dir,
                    NOISE_PERCENTAGE, NOISE_SHIFT, SOFT_PROB
                )
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss.item()} "
                  f"(Unary: {unary_loss.item()}, Pairwise: {pairwise_loss.item()})")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'Training Loss (Noise: {NOISE_PERCENTAGE*100:.1f}%, Shift: {NOISE_SHIFT}px)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete! Results saved to {output_dir}/")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()

