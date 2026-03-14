# SAM-CRF: High-Level Pairwise Regularization for Weakly Supervised Semantic Segmentation

## Overview

SAM-CRF trains with a conditional random field (CRF)-based loss: the per-pixel **unary** term encourages predictions to be similar to dino.txt pseudo-labels, and the **pairwise** term regularizes predictions using SAM boundaries. In terms of the architecture, SAM-CRF adds a lightweight decoder that runs parallel to the dino.txt vision head, both on top of a frozen DINOv3 backbone.

## Method

- **Pairwise term:** Pairwise affinities in the relaxed Potts loss are defined using SAM boundaries. **Dilation** is applied to account for the partial voluming effect in images.
- **Unary term:** Per-pixel loss uses **collision cross-entropy** (not standard cross-entropy) on soft pseudo-labels from dino.txt.
- **Decoder:** One transformer block + two convolution blocks are designed on top of a frozen DINO backbone; only the decoder is trained.

## Results

| Dataset           | mIoU   |
|-------------------|--------|
| PASCAL VOC 2012   | 80.6%  |
| MS COCO 2014      | 56.6%  |

---
We use the models DINOv3, dino.txt, and SAM in this work.

*This repository is currently under-documented. A more complete and readable version will be added later.*
