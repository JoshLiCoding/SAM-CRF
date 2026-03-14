import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(sam_contour, affinity='dilate', sigma=2.0):
    device = sam_contour.device
    sam_contour_cpu = sam_contour.cpu().numpy().astype(np.float32)

    if affinity == 'gaussian':
        softened = np.zeros_like(sam_contour_cpu)
        for i in range(sam_contour_cpu.shape[0]):
            # distance_transform_edt(x): dist from each non-zero to nearest zero. Invert so sources are 1s.
            dist = ndimage.distance_transform_edt(1.0 - sam_contour_cpu[i])
            softened[i] = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        w_np = 1.0 - softened
    else: 
        dilated_contour = np.zeros_like(sam_contour_cpu)
        for i in range(sam_contour_cpu.shape[0]):
            dilated_contour[i] = ndimage.maximum_filter(sam_contour_cpu[i], size=5)
        w_np = 1.0 - dilated_contour

    w = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
    return w

def CollisionCrossEntropyLoss(logits, target_probs, weight_by_foreground=False):
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
        foreground_weight = 1.0 - target_probs[:, 0:1, : , :]  # (B, 1, H, W)
        per_pixel_loss_weighted = per_pixel_loss * foreground_weight.squeeze(1)  # (B, H, W)
        loss = per_pixel_loss_weighted.sum() / (foreground_weight.sum() + 1e-8)
    else:
        loss = per_pixel_loss.mean()
    
    return loss


def CrossEntropyLoss(logits, target_probs, soft_targets=False):
    log_pred = F.log_softmax(logits, dim=1)
    if soft_targets:
        # CE = -sum_k target_k * log(pred_k)
        per_element = -torch.sum(target_probs * log_pred, dim=1)
    else:
        # Hard: one-hot from argmax, then CE = -log(pred[true_class])
        target_class = target_probs.argmax(dim=1)  # (B, ...)
        per_element = F.nll_loss(log_pred, target_class, reduction='none')
    return per_element.mean()

def ReverseCrossEntropyLoss(logits, target_probs, eps=1e-8):
    """
    Reverse Cross-Entropy: CE(Prediction, Target) = -sum(Prediction * log(Target))
    Flips the standard CE inputs.
    """
    pred = F.softmax(logits, dim=1)
    # Clamp to prevent log(0) if target_probs contains absolute zeros
    log_target = torch.log(target_probs.clamp(min=eps))
    
    per_element = -torch.sum(pred * log_target, dim=1)
    return per_element.mean()


def KLEntropyLoss(logits, target_probs):
    """
    KL(Target || Prediction) + Entropy(Prediction)
    """
    log_pred = F.log_softmax(logits, dim=1)
    # Reusing log_pred to compute pred is slightly more efficient than calling F.softmax again
    pred = torch.exp(log_pred) 
    
    # F.kl_div expects log-probabilities as input and probabilities as target.
    # reduction='none' allows us to sum over the class dimension (dim=1) safely.
    kl = F.kl_div(log_pred, target_probs, reduction='none').sum(dim=1)
    
    # Entropy of model prediction: -sum(p * log(p))
    entropy = -torch.sum(pred * log_pred, dim=1)
    
    return (kl + entropy).mean()


def KLDivergenceLoss(logits, target_probs, eps=1e-8):
    """
    Zero-avoiding KL divergence loss: KL(target || pred), with gradients w.r.t. logits.

    Uses log_softmax for numerical stability and clamps target_probs to avoid log(0).
    Suitable for training logits to match soft targets (e.g. pseudolabels).

    Args:
        logits: (B, C, ...) tensor of logits from the model (will be trained).
        target_probs: (B, C, ...) tensor of target probabilities (same shape as logits).
        eps: minimum value for target_probs to avoid log(0). Clamped values are
            renormalized so target remains a valid distribution over the class dim.

    Returns:
        Scalar loss: mean over batch and spatial dimensions of sum over classes
        of target_k * (log(target_k) - log(pred_k)).
    """
    # Numerically stable: log(pred) from logits without forming pred explicitly
    log_pred = F.log_softmax(logits, dim=1)

    # Zero-avoiding: clamp target so log(target) is always finite
    target_safe = target_probs.clamp(min=eps)
    # Renormalize over class dim so we still have a distribution
    target_safe = target_safe / target_safe.sum(dim=1, keepdim=True)

    # KL(target || pred) = sum_k target_k * (log(target_k) - log(pred_k))
    kl = target_safe * (torch.log(target_safe) - log_pred)
    loss = kl.sum(dim=1).mean()
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
        affinity: 'dilate' (default) = soft gaussian dilation; 'maximum' = hard max-filter dilation
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
        class_weights = torch.full((num_classes,), 75.0, device=device, dtype=prob.dtype)

        # dl=3 with balanced background unary loss for sam x2
        # class_weights[0]  = 0.0  # background (max mIoU: 0.9005)
        # class_weights[1]  = 900.0  # aeroplane (max mIoU: 0.8505)
        # class_weights[2]  = 0.0
        # class_weights[3]  = 200.0  # bird (max mIoU: 0.9225)
        # class_weights[4]  = 75.0   # boat (max mIoU: 0.7586)
        # class_weights[5]  = 75.0   # bottle (max mIoU: 0.8010)
        # class_weights[6]  = 300.0  # bus (max mIoU: 0.9423)
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8387) UPDATED
        # class_weights[8]  = 200.0  # cat (max mIoU: 0.9557)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4567)
        # class_weights[10] = 200.0  # cow (max mIoU: 0.9350)
        # class_weights[11] = 0.0   # diningtable
        # class_weights[12] = 300.0  # dog (max mIoU: 0.9420)
        # class_weights[13] = 300.0  # horse (max mIoU: 0.8979)
        # class_weights[14] = 200.0  # motorbike (max mIoU: 0.8384) UPDATED
        # class_weights[15] = 200.0  # person (max mIoU: 0.8731)
        # class_weights[16] = 0.0   # potted plant
        # class_weights[17] = 200.0  # sheep (max mIoU: 0.9286) UPDATED
        # class_weights[18] = 50.0   # sofa (max mIoU: 0.5832)
        # class_weights[19] = 200.0  # train (max mIoU: 0.9062) UPDATED
        # class_weights[20] = 50.0   # tv/monitor (max mIoU: 0.6771)

        # Coco weights
        # class_weights[0]  = 0.0   # background (max mIoU: 0.7964)
        # class_weights[1]  = 50.0    # person (max mIoU: 0.5149)
        # class_weights[2]  = 50.0    # bicycle (max mIoU: 0.5321)
        # class_weights[3]  = 50.0    # car (max mIoU: 0.5584)
        # class_weights[4]  = 300.0   # motorcycle (max mIoU: 0.6736)
        # class_weights[5]  = 500.0   # airplane (max mIoU: 0.2817)
        # class_weights[6]  = 500.0   # bus (max mIoU: 0.7829)
        # class_weights[7]  = 500.0   # train (max mIoU: 0.7804)
        # class_weights[8]  = 200.0   # truck (max mIoU: 0.5810)
        # class_weights[9]  = 100.0   # boat (max mIoU: 0.5646)
        # class_weights[10] = 75.0    # traffic light (max mIoU: 0.6023)
        # class_weights[11] = 200.0   # fire hydrant (max mIoU: 0.8254)
        # class_weights[12] = 500.0   # stop sign (max mIoU: 0.8335)
        # class_weights[13] = 200.0   # parking meter (max mIoU: 0.7530)
        # class_weights[14] = 200.0   # bench (max mIoU: 0.5009)
        # class_weights[15] = 300.0   # bird (max mIoU: 0.6245)
        # class_weights[16] = 500.0   # cat (max mIoU: 0.7852)
        # class_weights[17] = 500.0   # dog (max mIoU: 0.7369)
        # class_weights[18] = 300.0   # horse (max mIoU: 0.6802)
        # class_weights[19] = 200.0   # sheep (max mIoU: 0.7486)
        # class_weights[20] = 100.0   # cow (max mIoU: 0.7832)
        # class_weights[21] = 300.0   # elephant (max mIoU: 0.8335)
        # class_weights[22] = 200.0   # bear (max mIoU: 0.8493)
        # class_weights[23] = 75.0    # zebra (max mIoU: 0.7850)
        # class_weights[24] = 75.0    # giraffe (max mIoU: 0.7556)
        # class_weights[25] = 50.0    # backpack (max mIoU: 0.3969)
        # class_weights[26] = 100.0   # umbrella (max mIoU: 0.7751)
        # class_weights[27] = 0.0     # handbag (max mIoU: 0.3013)
        # class_weights[28] = 0.0     # tie (max mIoU: 0.3144)
        # class_weights[29] = 200.0   # suitcase (max mIoU: 0.6804)
        # class_weights[30] = 100.0   # frisbee (max mIoU: 0.6371)
        # class_weights[31] = 200.0   # skis (max mIoU: 0.1672)
        # class_weights[32] = 200.0   # snowboard (max mIoU: 0.5584)
        # class_weights[33] = 50.0    # sports ball (max mIoU: 0.3777)
        # class_weights[34] = 400.0   # kite (max mIoU: 0.5130)
        # class_weights[35] = 75.0    # baseball bat (max mIoU: 0.3747)
        # class_weights[36] = 200.0   # baseball glove (max mIoU: 0.4554)
        # class_weights[37] = 500.0   # skateboard (max mIoU: 0.4120)
        # class_weights[38] = 100.0   # surfboard (max mIoU: 0.6975)
        # class_weights[39] = 100.0   # tennis racket (max mIoU: 0.5182)
        # class_weights[40] = 0.0     # bottle (max mIoU: 0.3767)
        # class_weights[41] = 0.0     # wine glass (max mIoU: 0.4452)
        # class_weights[42] = 0.0     # cup (max mIoU: 0.4163)
        # class_weights[43] = 50.0    # fork (max mIoU: 0.3236)
        # class_weights[44] = 50.0    # knife (max mIoU: 0.4083)
        # class_weights[45] = 0.0     # spoon (max mIoU: 0.2810)
        # class_weights[46] = 0.0     # bowl (max mIoU: 0.2435)
        # class_weights[47] = 100.0   # banana (max mIoU: 0.7147)
        # class_weights[48] = 50.0    # apple (max mIoU: 0.6462)
        # class_weights[49] = 0.0     # sandwich (max mIoU: 0.2518)
        # class_weights[50] = 50.0    # orange (max mIoU: 0.7002)
        # class_weights[51] = 50.0    # broccoli (max mIoU: 0.5491)
        # class_weights[52] = 0.0     # carrot (max mIoU: 0.4782)
        # class_weights[53] = 50.0    # hot dog (max mIoU: 0.5034)
        # class_weights[54] = 200.0   # pizza (max mIoU: 0.6396)
        # class_weights[55] = 0.0     # donut (max mIoU: 0.4884)
        # class_weights[56] = 200.0   # cake (max mIoU: 0.4780)
        # class_weights[57] = 75.0    # chair (max mIoU: 0.4512)
        # class_weights[58] = 50.0    # couch (max mIoU: 0.5680)
        # class_weights[59] = 0.0     # potted plant (max mIoU: 0.1635)
        # class_weights[60] = 400.0   # bed (max mIoU: 0.6208)
        # class_weights[61] = 50.0    # dining table (max mIoU: 0.2072)
        # class_weights[62] = 500.0   # toilet (max mIoU: 0.7289)
        # class_weights[63] = 75.0    # tv (max mIoU: 0.3816)
        # class_weights[64] = 75.0    # laptop (max mIoU: 0.6335)
        # class_weights[65] = 400.0   # mouse (max mIoU: 0.5668)
        # class_weights[66] = 200.0   # remote (max mIoU: 0.5352)
        # class_weights[67] = 100.0   # keyboard (max mIoU: 0.6651)
        # class_weights[68] = 50.0    # cell phone (max mIoU: 0.6131)
        # class_weights[69] = 100.0   # microwave (max mIoU: 0.6574)
        # class_weights[70] = 100.0   # oven (max mIoU: 0.5387)
        # class_weights[71] = 50.0    # toaster (max mIoU: 0.3408)
        # class_weights[72] = 300.0   # sink (max mIoU: 0.3827)
        # class_weights[73] = 75.0    # refrigerator (max mIoU: 0.6817)
        # class_weights[74] = 0.0     # book (max mIoU: 0.3760)
        # class_weights[75] = 200.0   # clock (max mIoU: 0.5523)
        # class_weights[76] = 100.0   # vase (max mIoU: 0.3971)
        # class_weights[77] = 50.0    # scissors (max mIoU: 0.5934)
        # class_weights[78] = 100.0   # teddy bear (max mIoU: 0.7233)
        # class_weights[79] = 0.0     # hair drier (max mIoU: 0.4559)
        # class_weights[80] = 0.0     # toothbrush (max mIoU: 0.3441)

        # voc weights from kl divergence
        class_weights[0]  = 0.0  # background (max mIoU: 0.8929)
        class_weights[1]  = 1000.0 # aeroplane (max mIoU: 0.8335)
        class_weights[2]  = 0.0    # bicycle (max mIoU: 0.4772)
        class_weights[3]  = 700.0  # bird (max mIoU: 0.8787)
        class_weights[4]  = 300.0  # boat (max mIoU: 0.7342)
        class_weights[5]  = 100.0  # bottle (max mIoU: 0.5836)
        class_weights[6]  = 500.0  # bus (max mIoU: 0.9396)
        class_weights[7]  = 400.0  # car (max mIoU: 0.8419)
        class_weights[8]  = 300.0  # cat (max mIoU: 0.9360)
        class_weights[9]  = 200.0  # chair (max mIoU: 0.4530)
        class_weights[10] = 200.0  # cow (max mIoU: 0.9249)
        class_weights[11] = 0.0    # diningtable (max mIoU: 0.4489)
        class_weights[12] = 800.0  # dog (max mIoU: 0.9039)
        class_weights[13] = 700.0  # horse (max mIoU: 0.8704)
        class_weights[14] = 400.0  # motorbike (max mIoU: 0.8093)
        class_weights[15] = 300.0  # person (max mIoU: 0.8482)
        class_weights[16] = 100.0  # potted plant (max mIoU: 0.5221)
        class_weights[17] = 400.0  # sheep (max mIoU: 0.9096)
        class_weights[18] = 100.0  # sofa (max mIoU: 0.5234)
        class_weights[19] = 700.0  # train (max mIoU: 0.8674)
        class_weights[20] = 100.0  # tv/monitor (max mIoU: 0.5426)

        # voc weights from weighted cce, new version
        # class_weights[0]  = 0.0  # background (max mIoU: 0.9005)
        # class_weights[1]  = 1000.0  # aeroplane (max mIoU: 0.8505)
        # class_weights[2]  = 0.0
        # class_weights[3]  = 300.0  # bird (max mIoU: 0.9225)
        # class_weights[4]  = 100.0   # boat (max mIoU: 0.7586)
        # class_weights[5]  = 75.0   # bottle (max mIoU: 0.8010)
        # class_weights[6]  = 300.0  # bus (max mIoU: 0.9423)
        # class_weights[7]  = 200.0  # car (max mIoU: 0.8387) UPDATED
        # class_weights[8]  = 200.0  # cat (max mIoU: 0.9557)
        # class_weights[9]  = 50.0   # chair (max mIoU: 0.4567)
        # class_weights[10] = 200.0  # cow (max mIoU: 0.9350)
        # class_weights[11] = 0.0   # diningtable
        # class_weights[12] = 300.0  # dog (max mIoU: 0.9420)
        # class_weights[13] = 300.0  # horse (max mIoU: 0.8979)
        # class_weights[14] = 200.0  # motorbike (max mIoU: 0.8384) UPDATED
        # class_weights[15] = 200.0  # person (max mIoU: 0.8731)
        # class_weights[16] = 0.0   # potted plant
        # class_weights[17] = 200.0  # sheep (max mIoU: 0.9286) UPDATED
        # class_weights[18] = 50.0   # sofa (max mIoU: 0.5832)
        # class_weights[19] = 200.0  # train (max mIoU: 0.9062) UPDATED
        # class_weights[20] = 50.0   # tv/monitor (max mIoU: 0.6771)

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