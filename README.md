ANATOMICALLY RELEVANT FRAME SELECTION FOR LARYNGOSCOPY

This repository provides a framework for automatic selection of anatomically relevant frames in laryngoscopic videos. 
The system leverages DINOv3 as a feature extractor and an MLP classifier to distinguish between anatomically relevant frames and non-relevant ones.
The pipeline is divided into two main phases:
- Batch Inference: Extract embeddings using DINOv3 and classify frames with the MLP.
- Active Learning: Fine-tune the model on a new dataset to improve performance.

DIRECTORY STRUCTURE:

To ensure the script runs correctly, organize your local environment as follows:
- DATASET_FOLDER: Path to the raw frames extracted from the public dataset.
- WEIGHTS_PATH: Path to the pre-trained MLP weights (e.g., ./models/best_mlp_seed_99.pth).
- WORKSPACE_DIR: The output directory where the script will create:
    - /INF/: Frames classified as Informative.
    - /NON_INF/: Frames classified as Non-Informative.
    - embeddings_db.pt: Serialized DINOv3 features for Active Learning.
    - inference_log.csv: A summary of the initial predictions.

DATASET:

The framework has been designed to be compatible with the Laryngeal Endoscopy Dataset available on Zenodo.

Source: https://doi.org/10.5281/zenodo.1162784

Instructions: Extract frames from the provided videos and store them in a single folder before running the script.

MODEL BACKBONE:

The backbone used is DINOv3 (dinov3-vits16-pretrain-lvd1689m). 
The model is hosted on Hugging Face and will be downloaded automatically the first time you run the script.
