import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(sam_contour):
    device = sam_contour.device

    # dilate by 3
    sam_contour_cpu = sam_contour.cpu().numpy()
    dilated_contour = np.zeros_like(sam_contour_cpu)
    for i in range(sam_contour_cpu.shape[0]):
        dilated_contour[i] = ndimage.maximum_filter(sam_contour_cpu[i], size=3)
    sam_contour = torch.from_numpy(dilated_contour).to(device)

    w = (~sam_contour.bool()).to(torch.float32)
    return w

def CollisionCrossEntropyLoss(logits, target_probs, weight_by_foreground=True):
    """
    See "Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation" paper.
    CCE loss is robust to pseudo-label uncertainty without requiring hard labels.
    
    Args:
        logits: (B, C, H, W) tensor of logits from model
        target_probs: (B, C, H, W) tensor of target probabilities (pseudolabels)
        weight_by_foreground: If True, weight each pixel's loss by its foreground mass
            (1 - target_probs[:, 0]) and normalize by total foreground mass, so background-
            heavy pixels contribute less and foreground pixels contribute more (evenly).
    """
    probs = torch.softmax(logits, dim=1)
    
    # Compute sum_k(σ_i^k * y_i^k) for each pixel
    sum_probs_target = torch.sum(probs * target_probs, dim=1)  # (B, H, W)
    
    # Standard CCE: -ln(sum_k(σ_i^k * y_i^k))
    per_pixel_loss = -torch.log(sum_probs_target + 1e-8)  # (B, H, W)
    
    if weight_by_foreground:
        # Weight by foreground mass: background-heavy pixels contribute less
        foreground_weight = 1.0 - target_probs[:, 0:1, :, :]  # (B, 1, H, W)
        per_pixel_loss_weighted = per_pixel_loss * foreground_weight.squeeze(1)  # (B, H, W)
        loss = per_pixel_loss_weighted.sum() / (foreground_weight.sum() + 1e-8)
    else:
        loss = per_pixel_loss.mean()
    
    return loss

def PottsLoss(type, logits, sam_contours_x, sam_contours_y, use_color_diff=False):
    """
    Potts loss with optional color-difference based weighting.
    
    Args:
        type: Type of Potts loss ('bilinear', 'quadratic', 'log_quadratic')
        logits: (B, C, H, W) tensor of logits
        sam_contours_x: (B, H, W-1) tensor of horizontal contours/weights
        sam_contours_y: (B, H-1, W) tensor of vertical contours/weights
        use_color_diff: If True, use contours directly as weights (no negation)
    """
    if use_color_diff:
        # For color_diff, contours are already weights, use directly
        w_x = sam_contours_x
        w_y = sam_contours_y
    else:
        # For other methods, negate contours to get weights
        w_x = calculate_pairwise_affinity(sam_contours_x)
        w_y = calculate_pairwise_affinity(sam_contours_y)
    
    if type == 'bilinear':
        weighting = 200.0
        
        prob = torch.softmax(logits, dim=1)
        
        prob_x = torch.roll(prob, -1, dims=3)
        loss_x = 1 - torch.sum(prob*prob_x, dim=1)
        loss_x = loss_x[:, :, :-1] * w_x

        prob_y = torch.roll(prob, -1, dims=2)
        loss_y = 1 - torch.sum(prob*prob_y, dim=1)
        loss_y = loss_y[:, :-1, :] * w_y
    elif type == 'quadratic':
        prob = torch.softmax(logits, dim=1)
        num_classes = prob.shape[1]
        
        device = prob.device
        class_weights = torch.full((num_classes,), 100.0, device=device, dtype=prob.dtype)

        # dl=3
        # class_weights[0]  = 0.0  # background
        # class_weights[1]  = 400.0  # aeroplane (max mIoU: 0.8695)
        # class_weights[2]  = 100.0  # bicycle (max mIoU: 0.3435)
        # class_weights[3]  = 300.0  # bird (max mIoU: 0.9204) CHANGED
        # class_weights[4]  = 100.0   # boat (max mIoU: 0.7245) CHANGED
        # class_weights[5]  = 50.0   # bottle (max mIoU: 0.7887)
        # class_weights[6]  = 400.0  # bus (max mIoU: 0.9462)
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8650)
        # class_weights[8]  = 300.0  # cat (max mIoU: 0.9618)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4472)
        # class_weights[10] = 100.0  # cow (max mIoU: 0.9353)
        # class_weights[11] = 0.0    # diningtable (max mIoU: 0.2975)
        # class_weights[12] = 200.0  # dog (max mIoU: 0.9464)
        # class_weights[13] = 200.0  # horse (max mIoU: 0.8835)
        # class_weights[14] = 200.0  # motorbike (max mIoU: 0.8285)
        # class_weights[15] = 100.0  # person (max mIoU: 0.8654)
        # class_weights[16] = 0.0    # potted plant (max mIoU: 0.4653)
        # class_weights[17] = 200.0  # sheep (max mIoU: 0.9230)
        # class_weights[18] = 50.0   # sofa (max mIoU: 0.4997)
        # class_weights[19] = 100.0  # train (max mIoU: 0.8976)
        # class_weights[20] = 50.0   # tv/monitor (max mIoU: 0.4269)

        # dl=3 updated
        # class_weights[0]  = 0.0  # background
        # class_weights[1]  = 400.0  # aeroplane (max mIoU: 0.8695)
        # class_weights[2]  = 100.0  # bicycle (max mIoU: 0.3435)
        # class_weights[3]  = 100.0  # bird
        # class_weights[4]  = 100.0   # boat (max mIoU: 0.7245)
        # class_weights[5]  = 50.0   # bottle (max mIoU: 0.7887)
        # class_weights[6]  = 300.0  # bus
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8650)
        # class_weights[8]  = 300.0  # cat (max mIoU: 0.9618)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4472)
        # class_weights[10] = 100.0  # cow (max mIoU: 0.9353)
        # class_weights[11] = 0.0    # diningtable (max mIoU: 0.2975)
        # class_weights[12] = 200.0  # dog (max mIoU: 0.9464)
        # class_weights[13] = 200.0  # horse (max mIoU: 0.8835)
        # class_weights[14] = 200.0  # motorbike (max mIoU: 0.8285)
        # class_weights[15] = 100.0  # person (max mIoU: 0.8654)
        # class_weights[16] = 40.0    # potted plant (max mIoU: 0.4653)
        # class_weights[17] = 200.0  # sheep (max mIoU: 0.9230)
        # class_weights[18] = 50.0   # sofa (max mIoU: 0.4997)
        # class_weights[19] = 300.0  # train
        # class_weights[20] = 50.0   # tv/monitor (max mIoU: 0.4269)

        # dl=5
        # class_weights[0]  = 0.0  # background
        # class_weights[1]  = 400.0  # aeroplane (max mIoU: 0.8895 @ w=400)
        # class_weights[2]  = 100.0  # bicycle (max mIoU: 0.3654 @ w=100)
        # class_weights[3]  = 300.0  # bird (max mIoU: 0.9137 @ w=300)
        # class_weights[4]  = 100.0  # boat (max mIoU: 0.7708 @ w=100)
        # class_weights[5]  = 100.0  # bottle (max mIoU: 0.8162 @ w=100)
        # class_weights[6]  = 300.0  # bus (max mIoU: 0.9403 @ w=300)
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8794 @ w=200)
        # class_weights[8]  = 300.0  # cat (max mIoU: 0.9471 @ w=300)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4780 @ w=50)
        # class_weights[10] = 100.0  # cow (max mIoU: 0.9278 @ w=100)
        # class_weights[11] = 50.0   # diningtable (max mIoU: 0.3348 @ w=50)
        # class_weights[12] = 300.0  # dog (max mIoU: 0.9384 @ w=300)
        # class_weights[13] = 300.0  # horse (max mIoU: 0.8749 @ w=300)
        # class_weights[14] = 400.0  # motorbike (max mIoU: 0.8331 @ w=400)
        # class_weights[15] = 100.0  # person (max mIoU: 0.8601 @ w=100)
        # class_weights[16] = 50.0   # potted plant (max mIoU: 0.5051 @ w=50)
        # class_weights[17] = 500.0  # sheep (max mIoU: 0.9204 @ w=500)
        # class_weights[18] = 50.0   # sofa (max mIoU: 0.5157 @ w=50)
        # class_weights[19] = 200.0  # train (max mIoU: 0.9032 @ w=200)
        # class_weights[20] = 50.0   # tv/monitor (max mIoU: 0.5277 @ w=50)

        # dl=5 updated
        # class_weights[0]  = 0.0  # background
        # class_weights[1]  = 400.0  # aeroplane (max mIoU: 0.8895 @ w=400)
        # class_weights[2]  = 100.0  # bicycle (max mIoU: 0.3654 @ w=100)
        # class_weights[3]  = 300.0  # bird (max mIoU: 0.9137 @ w=300)
        # class_weights[4]  = 90.0  # boat (max mIoU: 0.7985 @ w=90)
        # class_weights[5]  = 100.0  # bottle (max mIoU: 0.8162 @ w=100)
        # class_weights[6]  = 300.0  # bus (max mIoU: 0.9403 @ w=300)
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8794 @ w=200)
        # class_weights[8]  = 300.0  # cat (max mIoU: 0.9471 @ w=300)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4780 @ w=50)
        # class_weights[10] = 100.0  # cow (max mIoU: 0.9278 @ w=100)
        # class_weights[11] = 10.0   # diningtable (max mIoU: 0.4615 @ w=10)
        # class_weights[12] = 300.0  # dog (max mIoU: 0.9384 @ w=300)
        # class_weights[13] = 300.0  # horse (max mIoU: 0.8749 @ w=300)
        # class_weights[14] = 400.0  # motorbike (max mIoU: 0.8331 @ w=400)
        # class_weights[15] = 100.0  # person (max mIoU: 0.8601 @ w=100)
        # class_weights[16] = 60.0   # potted plant (max mIoU: 0.5386 @ w=60)
        # class_weights[17] = 500.0  # sheep (max mIoU: 0.9204 @ w=500)
        # class_weights[18] = 10.0   # sofa (max mIoU: 0.5298 @ w=10)
        # class_weights[19] = 200.0  # train (max mIoU: 0.9032 @ w=200)
        # class_weights[20] = 60.0   # tv/monitor (max mIoU: 0.5996 @ w=60)

        class_weights = class_weights.view(1, num_classes, 1, 1)  # (1, C, 1, 1) for broadcasting

        prob_x = torch.roll(prob, -1, dims=3)
        # Compute per-class loss and apply class weights before summing
        loss_x_per_class = 0.5 * (prob - prob_x)**2  # (B, C, H, W)
        loss_x_weighted = loss_x_per_class * class_weights  # (B, C, H, W)
        loss_x = torch.sum(loss_x_weighted, dim=1)  # (B, H, W)
        loss_x = loss_x[:, :, :-1] * w_x

        prob_y = torch.roll(prob, -1, dims=2)
        # Compute per-class loss and apply class weights before summing
        loss_y_per_class = 0.5 * (prob - prob_y)**2  # (B, C, H, W)
        loss_y_weighted = loss_y_per_class * class_weights  # (B, C, H, W)
        loss_y = torch.sum(loss_y_weighted, dim=1)  # (B, H, W)
        loss_y = loss_y[:, :-1, :] * w_y
    elif type == 'log_quadratic':
        prob = torch.softmax(logits, dim=1)
        num_classes = prob.shape[1]

        device = prob.device
        class_weights = torch.full((num_classes,), 30000.0, device=device, dtype=prob.dtype)

        class_weights = class_weights.view(1, num_classes, 1, 1)  # (1, C, 1, 1) for broadcasting

        diff_x = prob - torch.roll(prob, -1, dims=3)
        diff_y = prob - torch.roll(prob, -1, dims=2)

        sq_norm_x = torch.sum((diff_x ** 2) * class_weights, dim=1)
        sq_norm_y = torch.sum((diff_y ** 2) * class_weights, dim=1)

        eps = 1e-6

        inside_x = torch.clamp(1.0 - 0.5 * sq_norm_x[:, :, :-1], min=eps)
        inside_y = torch.clamp(1.0 - 0.5 * sq_norm_y[:, :-1, :], min=eps)

        loss_x = -torch.log(inside_x) * w_x
        loss_y = -torch.log(inside_y) * w_y

    loss = loss_x.mean() + loss_y.mean()
    return loss