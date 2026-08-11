# Deep Learning-Based Gait Recognition Using CNN-LSTM and Human Pose Estimation

This repository presents an educational pose-based gait recognition pipeline that uses MediaPipe landmarks and a CNN-LSTM classifier to classify walking sequences from short videos.

## 1. Project Overview

This educational computer-vision project identifies an enrolled class from a walking video. It converts video frames into MediaPipe Pose landmarks, normalizes the skeletal representation, constructs a 30-frame sequence, and applies a CNN-LSTM classifier.

> **Demonstration only:** the included dataset is synthetic. Its results are not real-world biometric-identification results and must not be presented as such.

## 2. Motivation

Human gait is a behavioral biometric signal that can be studied without relying on facial appearance. This project investigates whether pose-derived skeletal features can represent spatial body relationships and temporal walking patterns for closed-set identity classification.

## Research Contribution

The project investigates a pose-based representation for gait recognition by combining spatial and temporal deep-learning methods.

The main technical components are:

- Human-pose-based gait representation using 33 MediaPipe landmarks
- Hip-centered and torso-scale normalization
- Fixed-length temporal sequence construction
- Conv1D layers for spatial joint-feature representation
- LSTM for temporal movement modeling
- Closed-set identity classification
- Quantitative evaluation using precision, recall, F1-score and confusion matrices
- Interactive visualization through Streamlit

## 3. System Architecture

`Walking video -> OpenCV -> MediaPipe Pose (33 landmarks) -> normalization -> 30-frame sequence -> Conv1D -> LSTM -> softmax class scores -> predicted class`

## 4. Technologies

| Technology | Purpose |
|---|---|
| Python | Application and model code |
| OpenCV | Video decoding and frame processing |
| MediaPipe Pose | 33 body-landmark extraction |
| NumPy | Sequence processing |
| TensorFlow/Keras | CNN-LSTM model |
| scikit-learn | Data splitting and evaluation metrics |
| Matplotlib / Seaborn | Training and evaluation figures |
| Streamlit | Interactive web interface |

## 5. Dataset

The included generator creates a synthetic demonstration dataset with 4 classes, 20 sequences per class, 30 frames per sequence, and 99 features per frame (33 landmarks x x/y/z). It supports a clone-install-generate-train-evaluate workflow without a private video dataset.

For research with real data, document the dataset source, license, subjects, sequences, preprocessing, and split protocol. Do not equate synthetic results with real-world generalization.

## 6. Data Preprocessing

Each detected frame is represented by 33 xyz landmarks. Landmarks are translated so the hip midpoint becomes the origin and scaled by torso length (hip center to shoulder center). Valid pose frames are uniformly resampled to a fixed 30-frame sequence.

## 7. MediaPipe Pose Extraction

MediaPipe Pose processes each RGB frame and returns 33 landmark coordinates. A sequence is accepted only when at least two frames have a detectable pose. The Streamlit app displays the final annotated pose frame to make extraction visible.

## 8. CNN-LSTM Architecture

| Component | Configuration |
|---|---|
| Input | 30 frames × 99 pose features |
| Spatial feature encoder | Conv1D (64) → MaxPooling → Conv1D (96) |
| Regularization | Dropout (0.25) |
| Temporal encoder | LSTM (64) |
| Classifier | Dropout (0.30) → Dense softmax |

The Conv1D layers learn local feature relationships while the LSTM models their movement over time.

## 9. Training Methodology

The demonstration uses sequence-level stratified splitting. Because the same enrolled identities may occur across the splits, the resulting evaluation measures closed-set classification performance rather than generalization to previously unseen subjects.

Early stopping monitors validation loss and restores the best model weights.

This is a closed-set identity classifier: every predicted identity must appear in training. Therefore this demonstration does **not** claim subject-disjoint or unseen-person generalization.

## 10. Evaluation Metrics

The evaluation script reports held-out accuracy, per-class precision, recall, F1 score, and a confusion matrix. It writes structured results to `outputs/metrics.json`.

## 11. Experimental Results

The repository includes a lightweight synthetic demonstration dataset and provides scripts to train and evaluate the baseline model. After running the training and evaluation commands, the outputs folder contains metrics and plots that summarize performance on the held-out split.

The current demo is intended for reproducibility and educational inspection rather than claiming real-world biometric accuracy.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CNN-LSTM | Pending evaluation | Pending evaluation | Pending evaluation | Pending evaluation |

## 12. Confusion Matrix

After evaluation, the script writes a confusion matrix to `outputs/confusion_matrix.png`. This figure shows which enrolled classes are most often confused by the model on the held-out synthetic sequences.

## 13. Training Curves

Training produces `outputs/accuracy_curve.png` and `outputs/loss_curve.png`. These plots help inspect convergence and possible overfitting during the demonstration run.

## 14. Streamlit Demo

The app is served at `http://localhost:8501`. Upload a clear, single-person walking video and select **Analyze gait**. The app shows the predicted enrolled class, its softmax model score, class-score chart, and a detected pose frame.

Softmax scores are not calibrated confidence estimates. Predictions for identities absent from training are not meaningful.

## 15. Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If TensorFlow is not already installed, the dependency list in `requirements.txt` will install it for the environment used by this project.

## 16. Usage

```powershell
python -m scripts.generate_synthetic_data
python -m training.train --epochs 35
python -m evaluation.evaluate
streamlit run app.py
```

## 17. Project Structure

```text
app.py                          Streamlit demo interface
preprocessing/                 video processing, pose extraction, and normalization
models/cnn_lstm.py             CNN-LSTM model definition
scripts/generate_synthetic_data.py
training/train.py              training pipeline and artifact export
evaluation/                    evaluation metrics and visualization
models_artifacts/              saved model and class label mapping
outputs/                       metrics, confusion matrix, and training plots
tests/                         unit tests for preprocessing utilities
```

## 18. Limitations

- Synthetic samples do not represent real gait variation.
- Pose extraction can fail with occlusion, poor video quality, unusual camera angles, or multiple people.
- Gait may change with clothing, shoes, carried objects, injury, speed, and environment.
- This closed-set classifier is not an open-set identity or authentication system.

## 19. Ethical Considerations

Gait is biometric information. Use this project only for education and research with informed consent, data minimization, privacy safeguards, fairness assessment, and compliance with applicable law. It is not intended for surveillance or standalone security decisions.

## 20. Future Work

- Evaluate on consented real-world datasets with documented subject-level protocols.
- Compare classical, CNN-only, LSTM-only, and CNN-LSTM baselines.
- Study cross-view robustness, occlusion handling, and open-set recognition.
- Explore graph neural networks and transformer-based temporal models.

## 21. License

This project is licensed under the [MIT License](LICENSE).
