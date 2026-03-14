"""
Test script to evaluate model performance on boundary misalignment.

Creates a synthetic square image, downsamples 4x, and trains with:
- Two unary losses from two differently shifted masks (configurable shifts)
Visualization: original image at full resolution; other panels (GT, pred, unary 1, unary 2) at 4x down.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add paths
DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
sys.path.append(DINOV3_LOCATION)

from model.dino import DinoWSSS
from utils.loss import CollisionCrossEntropyLoss
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


def shift_mask(mask, shift_down=1, shift_right=1):
    """
    Shift a mask down and right by specified amounts.
    Negative values shift the opposite way (up/left).

    Args:
        mask: numpy array [H, W] or torch.Tensor [H, W] with class indices
        shift_down: Number of pixels to shift down (positive = down, negative = up)
        shift_right: Number of pixels to shift right (positive = right, negative = left)

    Returns:
        shifted_mask: Same type as input, shifted mask [H, W]
    """
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask.copy()

    H, W = mask_np.shape
    shifted_mask = np.zeros_like(mask_np)

    # Positive: content moves down/right. Negative: content moves up/left.
    if shift_down >= 0 and shift_right >= 0:
        # Shift down and right: copy from top-left to lower-right
        new_h_start, new_w_start = shift_down, shift_right
        new_h_end, new_w_end = H, W
        old_h_start, old_w_start = 0, 0
        old_h_end, old_w_end = H - shift_down, W - shift_right
    elif shift_down <= 0 and shift_right <= 0:
        # Shift up and left: copy from lower-right to top-left
        new_h_start, new_w_start = 0, 0
        new_h_end, new_w_end = H + shift_down, W + shift_right
        old_h_start, old_w_start = -shift_down, -shift_right
        old_h_end, old_w_end = H, W

    if new_h_end > new_h_start and new_w_end > new_w_start and old_h_end > old_h_start and old_w_end > old_w_start:
        shifted_mask[new_h_start:new_h_end, new_w_start:new_w_end] = mask_np[old_h_start:old_h_end, old_w_start:old_w_end]

    if is_tensor:
        return torch.from_numpy(shifted_mask).to(mask.device).long()
    else:
        return shifted_mask.astype(mask.dtype)


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


def compute_miou(pred, gt, num_classes=2):
    """mIoU between predicted and ground truth masks. pred, gt: [H,W] int with values 0..num_classes-1."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    return np.mean(valid) if valid else float('nan')


def visualize_results(image_orig, mask_gt, pred_down, mask_unary_1, mask_unary_2,
                     epoch, output_dir, shift_unary_1, shift_unary_2):
    """Original image at full resolution; other panels at 4x down. Shows both unary masks."""
    os.makedirs(output_dir, exist_ok=True)
    pred_prob = torch.softmax(pred_down, dim=0).cpu().numpy()[1]
    to_np = lambda t: t.cpu().numpy() if isinstance(t, torch.Tensor) else t
    mask_unary_1_np = to_np(mask_unary_1) if hasattr(mask_unary_1, 'cpu') else mask_unary_1
    mask_unary_2_np = to_np(mask_unary_2) if hasattr(mask_unary_2, 'cpu') else mask_unary_2

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # Full-resolution original image
    axes[0, 0].imshow(np.array(image_orig))
    axes[0, 0].set_title('Image (full res)')
    axes[0, 1].imshow(mask_gt, cmap='gray', interpolation='nearest')
    axes[0, 1].set_title('GT mask')
    axes[0, 2].imshow(pred_prob, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0, 2].set_title(f'Pred (epoch {epoch})')
    # Unary 1 (shifted mask) + pred prob overlay
    axes[1, 0].imshow(mask_unary_1_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].imshow(pred_prob, cmap='viridis', alpha=0.6, vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title(f'Unary 1 {shift_unary_1} + pred overlay')
    # Unary 2 (shifted mask) + pred prob overlay
    axes[1, 1].imshow(mask_unary_2_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].imshow(pred_prob, cmap='viridis', alpha=0.6, vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_title(f'Unary 2 {shift_unary_2} + pred overlay')
    # GT + pred overlay
    axes[1, 2].imshow(mask_gt, cmap='gray', interpolation='nearest')
    axes[1, 2].imshow(pred_prob, cmap='viridis', alpha=0.6, vmin=0, vmax=1, interpolation='nearest')
    axes[1, 2].set_title('GT + pred overlay')
    plt.suptitle(f'Epoch {epoch} | unary1 {shift_unary_1} | unary2 {shift_unary_2}')
    plt.tight_layout()
    path = os.path.join(output_dir, f'boundary_test_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SHIFT_UNARY_1 = (5, 5)
    SHIFT_UNARY_2 = (-5, -5)   # Second unary: different mask shift
    SOFT_PROB = 0.75
    NUM_EPOCHS = 1500
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (512, 512)
    SQUARE_SIZE = 256
    SQUARE_POS = (128, 128)
    # Model (no config file)
    backbone_name = 'dinov3_vitl16'
    num_transformer_blocks = 0
    num_conv_blocks = 1
    use_bottleneck = False
    use_transpose_conv = False

    image_orig, mask_orig = create_synthetic_square_image(
        size=IMAGE_SIZE, square_size=SQUARE_SIZE, square_pos=SQUARE_POS,
        color1=(255, 0, 0), color2=(0, 255, 0))
    image_down, mask_down = downsample_image_and_mask(image_orig, mask_orig, factor=4)
    H_down, W_down = mask_down.shape

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_image, _ = create_synthetic_dataset(image_orig, mask_orig, transform)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    target = torch.from_numpy(mask_down).long().to(device)
    output_dir = 'output_boundary_test'
    os.makedirs(output_dir, exist_ok=True)

    model = DinoWSSS(
        backbone_name=backbone_name,
        num_transformer_blocks=num_transformer_blocks,
        num_conv_blocks=num_conv_blocks,
        out_channels=2,
        use_bottleneck=use_bottleneck,
        use_transpose_conv=use_transpose_conv
    ).to(device)

    # t-SNE of backbone patch features (once)
    with torch.no_grad():
        _, patch_tokens, _ = model.get_backbone_features(transformed_image)
    X = patch_tokens[0].cpu().numpy()  # (P, D)
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, random_state=0).fit_transform(X)
    mask_unary_1_tsne = shift_mask(mask_down, shift_down=SHIFT_UNARY_1[0], shift_right=SHIFT_UNARY_1[1])
    p = int(np.sqrt(len(X)))
    mask_patch = F.interpolate(
        torch.from_numpy(mask_unary_1_tsne).float().unsqueeze(0).unsqueeze(0),
        size=(p, p), mode='nearest'
    ).squeeze().numpy().flatten()
    plt.figure(figsize=(6, 5))
    plt.scatter(xy[:, 0], xy[:, 1], c=mask_patch, cmap='coolwarm', s=8, alpha=0.8)
    plt.colorbar(label='Unary 1 (0=bg, 1=fg)')
    plt.title('Backbone patch features (t-SNE)')
    plt.savefig(os.path.join(output_dir, 'backbone_tsne.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_dir}/backbone_tsne.png")

    optimizer_params = [
        {'params': model.transformer_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.ln.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.lin_classifier.parameters(), 'lr': LEARNING_RATE},
    ]
    if use_transpose_conv:
        optimizer_params.append({'params': model.upsample_conv1.parameters(), 'lr': LEARNING_RATE})
        optimizer_params.append({'params': model.upsample_conv2.parameters(), 'lr': LEARNING_RATE})
    optimizer = torch.optim.SGD(optimizer_params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)

    losses = []
    losses_unary_1 = []
    losses_unary_2 = []
    miou_epochs = []
    miou_values = []
    pred_prob_frames = []  # for video: (epoch, pred_prob array)

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        model.backbone.eval()
        model.dinotxt_head.eval()
        optimizer.zero_grad()

        segmentations = model(transformed_image)['seg']
        B, C, H_seg, W_seg = segmentations.shape

        mask_unary_1 = shift_mask(target, shift_down=SHIFT_UNARY_1[0], shift_right=SHIFT_UNARY_1[1])
        is_fg_1 = (mask_unary_1 == 1).float()
        pseudolabel_probs_1 = torch.zeros((B, 2, H_seg, W_seg), dtype=torch.float32, device=device)
        pseudolabel_probs_1[:, 0] = is_fg_1 * (1.0 - SOFT_PROB) + (1.0 - is_fg_1) * SOFT_PROB
        pseudolabel_probs_1[:, 1] = is_fg_1 * SOFT_PROB + (1.0 - is_fg_1) * (1.0 - SOFT_PROB)

        mask_unary_2 = shift_mask(target, shift_down=SHIFT_UNARY_2[0], shift_right=SHIFT_UNARY_2[1])
        is_fg_2 = (mask_unary_2 == 1).float()
        pseudolabel_probs_2 = torch.zeros((B, 2, H_seg, W_seg), dtype=torch.float32, device=device)
        pseudolabel_probs_2[:, 0] = is_fg_2 * (1.0 - SOFT_PROB) + (1.0 - is_fg_2) * SOFT_PROB
        pseudolabel_probs_2[:, 1] = is_fg_2 * SOFT_PROB + (1.0 - is_fg_2) * (1.0 - SOFT_PROB)

        unary_loss_1 = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs_1)
        unary_loss_2 = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs_2)
        total_loss = unary_loss_1 + unary_loss_2
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        losses_unary_1.append(unary_loss_1.item())
        losses_unary_2.append(unary_loss_2.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred_down = model(transformed_image)['seg'][0]
                pred_prob = torch.softmax(pred_down, dim=0).cpu().numpy()[1]
                pred_mask = (pred_prob >= 0.5).astype(np.uint8)
                miou = compute_miou(pred_mask, mask_down, num_classes=2)
                miou_epochs.append(epoch + 1)
                miou_values.append(miou)
                pred_prob_frames.append((epoch + 1, pred_prob))
                mask_unary_1_np = shift_mask(target.cpu().numpy(), shift_down=SHIFT_UNARY_1[0], shift_right=SHIFT_UNARY_1[1])
                mask_unary_2_np = shift_mask(target.cpu().numpy(), shift_down=SHIFT_UNARY_2[0], shift_right=SHIFT_UNARY_2[1])
                visualize_results(
                    image_orig, mask_down, pred_down,
                    mask_unary_1_np, mask_unary_2_np, epoch + 1, output_dir,
                    SHIFT_UNARY_1, SHIFT_UNARY_2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(losses, label='Total')
    ax1.plot(losses_unary_1, label='Unary 1')
    ax1.plot(losses_unary_2, label='Unary 2')
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(miou_epochs, miou_values, 'b.-')
    ax2.set_title('Pred vs GT mIoU')
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=100, bbox_inches='tight')
    plt.close()
    final_miou = miou_values[-1] if miou_values else float('nan')
    print(f"Done. Results in {output_dir}/, final loss total={losses[-1]:.4f} unary1={losses_unary_1[-1]:.4f} unary2={losses_unary_2[-1]:.4f} mIoU={final_miou:.4f}" if not np.isnan(final_miou) else f"Done. Results in {output_dir}/, final loss total={losses[-1]:.4f} unary1={losses_unary_1[-1]:.4f} unary2={losses_unary_2[-1]:.4f}")

    # Compile pred_prob frames into video (gray heatmap, upscaled for readability)
    import imageio
    video_scale = 4  # upscale pred_prob to larger size
    video_fps = 20
    grays = np.clip(np.stack([p for _, p in pred_prob_frames]) * 255, 0, 255).astype(np.uint8)
    frames_rgb = np.stack([np.stack([g, g, g], axis=-1) for g in grays])
    h, w = frames_rgb[0].shape[:2]
    h_out, w_out = h * video_scale, w * video_scale
    frames_upscaled = [np.array(Image.fromarray(f).resize((w_out, h_out), Image.Resampling.LANCZOS)) for f in frames_rgb]
    out_path = os.path.join(output_dir, 'predictions.mp4')
    writer = imageio.get_writer(out_path, format='FFMPEG', fps=video_fps, codec='libx264', quality=8)
    for img in frames_upscaled:
        writer.append_data(img)
    writer.close()
    print(f"Saved video: {out_path} ({len(pred_prob_frames)} pred_prob frames)")


if __name__ == "__main__":
    main()

