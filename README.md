## Project Overview

This repository provides the official implementation of **UC-TwinLiteFormer**, a lightweight dual-task semantic segmentation framework, together with a geometry-constrained navigation line extraction algorithm (**GCP-NLE**), designed for real-time autonomous navigation of agricultural robots in under-canopy maize environments.

The framework targets challenging unstructured field conditions characterized by:

- Narrow inter-row corridors  
- Severe illumination variations  
- Heavy plant occlusion  
- Diverse obstacles (human, rock, ditch)

The proposed system integrates semantic perception and geometric modeling to generate stable and safe navigation lines under different traversability scenarios.

---

## Task Definition

The navigation perception problem is formulated as a **dual-task semantic segmentation task**:

- **Drivable road region segmentation**  
  - Extracts the traversable inter-row corridor.

- **Obstacle segmentation**  
  - Detects and classifies obstacles (human, rock, ditch) that may affect path feasibility.

The outputs of these two tasks jointly serve as the semantic foundation for downstream navigation line extraction.

---

## Model: UC-TwinLiteFormer

UC-TwinLiteFormer is a lightweight encoder–decoder network with only **1.46M parameters**, specifically optimized for under-canopy agricultural scenes.

### Key architectural features

- **Lightweight CNN–Transformer hybrid encoder**
  - Based on ESPNet-style efficient spatial pyramids
  - Enhanced with depthwise separable dilated convolutions (DESP)
  - Embedded lightweight Swin Transformer blocks for global context modeling

- **Dual-task decoder**
  - Road segmentation branch
  - Obstacle segmentation branch
  - Joint optimization enables task collaboration rather than independent perception

- **Feature fusion modules**
  - **STF (Skip-Transformer Fusion)**  
    Aligns shallow CNN features with deep Transformer features to balance spatial detail and semantic consistency.
  - **LAFAM (Lightweight Adaptive Feature Alignment Module)**  
    Performs channel- and spatial-level adaptive weighting for multi-scale feature alignment.

### Training strategy

- Composite loss: **Focal Loss + Tversky Loss + Dice Loss**
- Different loss parameterization for road and obstacle branches
- EMA-based parameter smoothing for stable convergence

---

## Navigation Line Extraction: GCP-NLE

To convert semantic segmentation results into executable navigation paths, this repository implements **GCP-NLE (Geometry-Constrained Polygonal Navigation Line Extraction)**.

### Core idea

Instead of relying on simple centerline or skeleton extraction, GCP-NLE:

- Explicitly models road boundaries and obstacle geometry using polygon representations
- Applies topology-aware geometric constraints
- Generates continuous, safe, and interpretable navigation lines

### Supported scenarios

GCP-NLE automatically adapts to three representative field conditions:

- **Obstacle-free corridor**
  - Navigation line generated from the geometric center of the road polygon.

- **Non-bypassable obstacle**
  - Obstacle fully blocks the corridor.
  - Navigation line is truncated before the obstacle using geometric constraints to prevent collision.

- **Bypassable obstacle**
  - Obstacle partially intrudes into the corridor.
  - A piecewise navigation line is generated:
    - Offset guidance  
    - Bypass traversal  
    - Smooth return to corridor center  

This design ensures path continuity while avoiding complex curve fitting, making the method efficient and suitable for real-time deployment.

---

## Key Advantages

- Unified modeling of road perception + obstacle awareness  
- Lightweight and real-time capable  
- Geometry-aware navigation that explicitly avoids obstacles  
- Interpretable and modular design for downstream planning and control  

---

## Reference

If you use this code or find it helpful, please cite the corresponding paper:

**A Lightweight Dual-Task Transformer Framework with Geometry-Constrained Path Extraction for Autonomous Under-Canopy Navigation in Maize Fields**
