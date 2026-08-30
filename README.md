# Real-Time Pose-Based Gait Recognition

An educational computer-vision project for closed-set gait classification from pose sequences. It supports recorded-video analysis and a live browser-camera dashboard with multi-person tracking, target selection, movement estimates, and robotics-ready target state.

> The included data is synthetic. This project is for education and consented research only; it is not a real-world biometric identification, surveillance, authentication, clinical, or safety-critical system.

## Features

- MediaPipe Pose extraction with 33 body landmarks
- Hip-centred, torso-scale skeletal normalization
- Fixed 30-frame gait sequences and CNN-LSTM recognition
- Live browser camera and recorded-video workflows
- OpenCV person detection and persistent tracking IDs
- Independent pose buffers and predictions for multiple people
- Configurable low-confidence `UNKNOWN` state
- Target-person selection and highlighted target tracking
- Image-space position, direction, and walking-speed estimates
- Gait-cycle estimates: cadence, cycle duration, ankle-separation peaks
- Detection/pose timing, FPS, and memory monitoring
- Training augmentation: small rotations, scaling, noise, and temporal jitter
- CNN-only, LSTM-only, and CNN-LSTM comparison experiment
- Controlled robustness evaluation
- Image-to-ground-plane calibration helper for configured cameras

## Architecture

```text
Camera or video
  -> Person detection
  -> Persistent tracking IDs
  -> MediaPipe Pose (33 landmarks per person)
  -> Normalization
  -> Per-person 30-frame buffer
  -> CNN / LSTM / CNN-LSTM classification
  -> Known identity or UNKNOWN
  -> Target state for robotics integration
```

The live baseline uses OpenCV's CPU HOG person detector and a centroid/IoU tracker. It is lightweight and useful for a demonstration, but it can miss people or switch IDs during crossings and long occlusions. Use a validated detector and tracker before deployment.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.download_pose_model
```

## Quick start

```powershell
python -m scripts.generate_synthetic_data
python -m training.train --epochs 35
python -m evaluation.evaluate
streamlit run app.py
```

Open the local address printed by Streamlit, choose **Live camera** or **Recorded video**, and grant browser camera permission for the live workflow.

`scripts.download_pose_model` retrieves the official lightweight MediaPipe Pose Landmarker asset required by the current MediaPipe Tasks API. It is not stored in Git.

## Live dashboard

The live workflow shows each tracked person’s ID, gait class, confidence, movement direction, image-space speed, FPS, detection time, pose time, and memory use.

Set a tracking ID as the target. The selected person is highlighted in green. The target state is structured as:

```text
id: person_02
gait: Person 2 | UNKNOWN
confidence: 0.94
position: [423, 287]
direction: RIGHT
speed_px_s: 82.4
tracking: true
gait_cycle: { cadence_steps_min, cycle_duration_s, ... }
```

`position` and `speed_px_s` are image-space quantities. A calibrated camera homography or depth sensor is required before interpreting them as metres or metres/second.

## Training and evaluation

The synthetic generator produces four classes, 20 sequences per class, 30 frames per sequence, and 99 features per frame (33 landmarks × xyz coordinates).

Train with augmentation applied only to the training partition:

```powershell
python -m training.train --epochs 35 --augment-copies 1
```

Evaluate the saved CNN-LSTM model:

```powershell
python -m evaluation.evaluate
```

This writes held-out accuracy, per-class precision/recall/F1, a confusion matrix, and training curves into `outputs/`.

Compare model architectures:

```powershell
python -m evaluation.compare_models --epochs 20 --augment-copies 1
```

This saves CNN-only, LSTM-only, and CNN-LSTM accuracy, weighted precision/recall/F1, and average inference time to `outputs/model_comparison.json`.

Run controlled robustness checks:

```powershell
python -m evaluation.robustness
```

The robustness script measures the held-out model under small pose noise and temporal jitter. It does not replace real-world testing across lighting, clothing, cameras, backgrounds, occlusion, distance, and walking direction.

## Calibration and real-world data

`preprocessing/calibration.py` converts a pixel point to a ground-plane coordinate using a supplied 3×3 camera homography. Calibrate and validate the camera in the actual environment before producing physical coordinates or walking speeds.

For real-world evaluation, use only consented data. Document dataset source and licence, participant consent, camera setup, sequences, lighting, clothing, speeds, distances, occlusion, retention, and split protocol. Use subject-disjoint splits when measuring generalization to previously unseen people.

## Project boundary: robotics-ready, not robotics

This project ends at the perception boundary. It produces person identity/status, tracking, image-space position, direction, speed, and gait-cycle estimates for a selected target. It does **not** contain a robot, ROS 2 nodes, mapping, localization, route/path planning, obstacle avoidance, motor control, or robot simulation.

```text
This project: camera -> perception -> robotics-ready target state

Future robotics project: target state -> ROS 2 -> planner -> obstacle avoidance
                       -> robot controller -> simulated or physical robot
```

Keep the future robotics integration as a separate project with its own safety requirements, interfaces, simulation tests, and deployment validation.

## Validation and freeze checklist

The repository is ready to freeze as an educational gait-perception baseline after completing the following checks for the intended release:

- `python -m pytest tests -q` passes.
- Synthetic data generation, training, evaluation, robustness, and model-comparison commands run successfully.
- The Streamlit dashboard opens and both tabs render.
- A consented camera/video test confirms expected behaviour for detection, tracking IDs, target highlighting, `UNKNOWN` thresholding, and output overlays.
- Results are labelled as synthetic or consented real-world results, with no unsupported biometric or robotics claims.
- The source version, dependency versions, dataset metadata, configuration, and generated metrics are recorded.

The current automated validation covers code paths and synthetic artefacts. Live-camera behaviour still requires manual validation on the machine/browser/camera used for deployment.

## Tests

```powershell
python -m pytest tests -q
```

Tests cover normalization, tracking-ID persistence, unknown handling, position/movement state, augmentation, gait-cycle calculations, calibration, and performance monitoring.

## Project structure

```text
app.py                          Streamlit dashboard
preprocessing/
  pose_extractor.py             MediaPipe pose extraction
  video_processor.py            recorded-video sequence construction
  normalization.py              skeletal normalization
  realtime.py                   tracking, target state, motion
  augmentation.py               training-only pose augmentation
  gait_cycle.py                 exploratory gait-cycle measures
  calibration.py                calibrated ground-plane projection
  performance.py                timing and memory monitor
models/cnn_lstm.py              CNN, LSTM, and CNN-LSTM definitions
training/train.py               training and artifact export
evaluation/                     evaluation, robustness, and comparison scripts
tests/                          automated unit tests
```

## Limitations and ethics

- Softmax scores are not calibrated confidence probabilities.
- The model is closed-set: it only recognizes identities represented during training. The `UNKNOWN` state is threshold-based, not a true open-set biometric solution.
- Synthetic scores must not be reported as real-world identity performance.
- Pose quality depends on camera angle, lighting, occlusion, distance, and multiple people.
- Gait is sensitive biometric information. Use informed consent, data minimization, privacy safeguards, fairness evaluation, and applicable legal compliance.

## Licence

This project is licensed under the [MIT License](LICENSE).
