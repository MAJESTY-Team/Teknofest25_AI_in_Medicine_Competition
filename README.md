# AI-Based Stroke Detection and Temporal Classification using Multimodal Imaging

## Project Overview

This research project, developed for the TEKNOFEST 2025 AI in Healthcare Competition, focuses on two critical tasks in neuroimaging: **Ischemic Stroke Detection from CT Scans** and **Temporal Classification of Stroke from Diffusion MRI**. The project leverages state-of-the-art computer vision architectures (YOLOv8, YOLOv11), investigates the efficacy of Vision Transformers (ViT), and introduces a novel hierarchical decision mechanism supported by Large Language Models (LLMs) for clinical explainability.

## 1. Experimental Methodologies

### Task A: Stroke Detection from CT Images
* **Objective:** Binary classification of brain CT images to detect the presence of ischemic stroke.
* **Data Strategy:**
    * Utilized the "Brain Stroke CT Image Dataset" (2,501 images) for training and the TEKNOFEST 2021 dataset for external validation.
    * Applied extensive data augmentation (rotation, brightness/contrast adjustment, Gaussian noise) to mitigate overfitting on limited data.
* **Architecture Comparison:**
    * Classic CNNs (EfficientNetB0, VGG16, Inception ResNetV2) were compared against Object Detection-based classifiers (YOLOv5, YOLOv8).
    * **Selected Model:** **YOLOv8** was selected as the final model due to superior F1 scores (0.9374) and inference speed compared to traditional CNNs.
* **Global Classification Technique:** Instead of segmenting specific lesions, the entire CT slice was treated as a single bounding box (coordinates 0.5, 0.5, 1.0, 1.0), effectively repurposing the YOLO object detector for global image classification to leverage its feature extraction capabilities.

### Task B: Temporal Classification from Diffusion MRI
* **Objective:** Classifying stroke onset time (e.g., Hyperacute, Subacute, Normal) using Diffusion-Weighted Imaging (DWI) and ADC maps.
* **Data Engineering:**
    * **Skull Stripping:** Implemented HD-BET for skull stripping to remove non-brain tissues and improve model focus.
    * **Synthetic Data Generation (Experimental):** Explored generative diffusion models (Stable Diffusion 1.5/XL, Flux.1, Prompt2MedImage) for data augmentation. While visually coherent, these were excluded from the final training pipeline due to risks of hallucination and clinical inconsistency.
* **Architectural Evolution:**
    * Experiments with Google Vision Transformer (ViT) showed convergence issues (training from scratch) or lower accuracy compared to CNN-based approaches.
    * **Selected Model:** **YOLOv11-medium (cls)** was chosen for its balance of high accuracy and computational efficiency.

## 2. Advanced Machine Learning Techniques

### Hierarchical "Two-Stage" Decision Mechanism
To address edge cases and low-confidence predictions, we developed a cascaded inference system :

1.  **Primary Inference:** The main YOLOv11 model classifies the MRI slice.
2.  **Confidence Check:** If the prediction confidence is below a specific threshold, the sample is not classified immediately.
3.  **Specialized Expert Models:** The low-confidence sample is routed to a specific binary classifier (e.g., *Hyperacute vs. Subacute*) trained exclusively to distinguish between those two confusing classes.
4.  **Result:** This method significantly improved the confusion matrix diagonal, specifically for temporally adjacent stroke stages.

### Error-Focused Training Loop
We implemented an iterative "human-in-the-loop" style data augmentation strategy:
1.  Train the initial model.
2.  Run inference on the training set to identify misclassified samples.
3.  Apply aggressive augmentation (mirroring, contrast shifts) *only* to these error-prone samples.
4.  Retrain the model with the enriched dataset to force the network to learn difficult features.

### Explainable AI (XAI) with MedGemma
To solve the "black box" problem in medical AI, we integrated **Google MedGemma** (a medical-specific LLM).
* **Workflow:** After the vision model establishes a diagnosis, the image and the label are fed into MedGemma.
* **Output:** The LLM generates a clinical report explaining *why* the diagnosis was made, highlighting relevant radiological features. This acts as a support layer for clinical interpretation.

## 3. Results Summary

| Task | Model | Metric | Score |
| :--- | :--- | :--- | :--- |
| **CT Stroke Detection** | YOLOv8 | F1 Score | **0.937**  |
| **CT Stroke Detection** | EfficientNetB0 | F1 Score | ]0.788  |
| **MRI Classification** | YOLOv11-m (Two-Stage) | Micro F1 | **0.866**  |
| **MRI Classification** | YOLOv11-m (Base) | Micro F1 | 0.854 |
| **MRI Classification** | Google ViT | Micro F1 | 0.627 |

## 4. Tools & Frameworks
* **Deep Learning:** PyTorch, Ultralytics YOLO (v8, v11), TensorFlow/Keras.
* **Preprocessing:** HD-BET (Skull Stripping), NumPy, Pandas.
* **Generative AI:** Stable Diffusion, Flux.1, MedGemma (LLM).
* **Hardware:** Training conducted on NVIDIA GPUs (RTX 3060/A100 equivalents implied by competition standards).
