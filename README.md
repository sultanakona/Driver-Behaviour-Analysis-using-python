# Driver Behaviour Analysis using Python

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](#)

A practical toolkit for detecting, analyzing, and visualizing driver behaviour using computer vision and machine learning. This repository combines video processing, pose and face analysis, and classification models to identify risky or distracted driving patterns and produce clear visual summaries.

<!-- Add a demo GIF or image here -->
![Demo](docs/demo-placeholder.gif)

Table of contents
- [Why this project?](#why-this-project)
- [Highlights](#highlights)
- [Features](#features)
- [Quickstart](#quickstart)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Run a demo](#run-a-demo)
- [How it works](#how-it-works)
- [Usage examples](#usage-examples)
- [Data & Models](#data--models)
- [Results & Visuals](#results--visuals)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

Why this project?
-----------------
Road safety is a global priority. Detecting dangerous driver behaviours like phone use, yawning, nodding off, or looking away can improve safety systems, enable alerts, and support research. This repo demonstrates a full pipeline from video -> features -> behaviour classification -> visual report, with code designed for clarity and experimentation.

Highlights
----------
- End-to-end pipeline: video preprocessing → landmark extraction → feature engineering → classification
- Supports real-time and batch processing modes
- Visual overlays and summary reports for easy interpretation
- Modular: swap detectors, trackers, and models with minimal changes
- Example Jupyter notebooks for exploration and reproducibility

Features
--------
- Face detection & alignment (e.g., MTCNN / OpenCV DNN)
- Facial landmark extraction (drowsiness, gaze estimation)
- Head pose estimation (to track attention)
- Hand/phone detection (YOLO / MobileNet-SSD)
- Behaviour classification (SVM / Random Forest / LightGBM / lightweight deep nets)
- Video annotation & frame-by-frame logging
- Performance metrics & confusion matrix generation

Quickstart
----------
Requirements
- Python 3.8+
- GPU recommended for model training and real-time inference (but not required for small demos)
- Common libraries: numpy, pandas, opencv-python, matplotlib, scikit-learn, imutils, torch/torchvision or tensorflow (depending on chosen models)

Install
1. Clone the repo
   ```bash
   git clone https://github.com/SabihaMishu/Driver-Behaviour-Analysis-using-Python.git
   cd Driver-Behaviour-Analysis-using-Python
   ```
2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate.bat       # Windows
   pip install -r requirements.txt
   ```
3. (Optional) Download pretrained model weights and datasets (see Data & Models below) and place them in the `weights/` and `data/` folders respectively.

Run a demo
- Run the example notebook to try the pipeline on a sample video:
  ```bash
  jupyter notebook notebooks/Demo.ipynb
  ```
- Quick inference script example:
  ```bash
  python src/infer.py --video sample_videos/driver1.mp4 --output outputs/annotated_driver1.mp4
  ```

How it works
------------
1. Video frames are captured and (optionally) resized for speed/throughput.
2. Face detector identifies the driver's face; facial landmarks and head pose are estimated.
3. Additional object detectors locate hands and phones.
4. Temporal features (blink rate, eye closure time, head pose variance, phone proximity) are computed.
5. A classifier predicts behaviours (e.g., attentive, distracted, texting, drowsy).
6. An annotated video and a per-video summary report are generated.

Usage examples
--------------
- Real-time monitoring (webcam):
  ```bash
  python src/realtime_monitor.py --camera 0
  ```
- Batch inference over a directory of dashcam clips:
  ```bash
  python src/batch_infer.py --input data/dashcam/ --output reports/
  ```
- Train a classifier with extracted features:
  ```bash
  python src/train_classifier.py --features data/features.csv --model outputs/model.pkl
  ```

Data & Models
-------------
- Example datasets: publicly available driver monitoring datasets (e.g., State Farm Distracted Driver, DriSE, AUC Drowsiness datasets). Always follow the dataset license and citation requirements.
- Preprocessing scripts in `src/data_preparation/` convert labeled videos into frame-level features.
- Pretrained weights (if included) go in the `weights/` folder. If you'd like pretrained artifacts added to the repo, open an issue or PR.

Results & Visuals
-----------------
- The repo contains example outputs in `outputs/` and visualizations in `docs/`:
  - Annotated videos with overlays (landmarks, bounding boxes, labels)
  - Frame-level CSV logs with timestamps and predicted classes
  - Confusion matrices and ROC curves for trained models
- Example performance numbers (illustrative — results depend on dataset and model choices):
  - Accuracy: 85–95% on curated test sets
  - Real-time throughput: ~10–30 FPS on a mid-range GPU (depends on model)

Best practices & tips
---------------------
- Use temporal smoothing (sliding windows) to reduce flicker in predictions.
- Combine multiple signals (head pose + eye closure + hand presence) for robust detection.
- Calibrate thresholds per-camera and per-environment (lighting, camera angle).
- When collecting new data, ensure diverse drivers, angles, and lighting conditions.

Contributing
------------
Contributions are welcome! Suggested ways to help:
- Add new detectors or lightweight model backbones
- Improve pre- and post-processing for robustness under low light
- Add more notebooks demonstrating evaluation and error analysis
- Add CI to verify notebooks build and tests pass

To contribute:
1. Fork the repo
2. Create a feature branch (e.g., `feat/phone-detector`)
3. Open a pull request describing your changes

License
-------
This project is provided under the MIT License. See the LICENSE file for details.

Contact
-------
Created by SabihaMishu — feel free to open issues, suggestions, or PRs on GitHub:
- Repository: [https://github.com/SabihaMishu/Driver-Behaviour-Analysis-using-Python](https://github.com/SabihaMishu/Driver-Behaviour-Analysis-using-Python)

Acknowledgements
----------------
- Thank you to the open-source computer vision and ML communities — prebuilt detectors, model architectures, and datasets made this possible.
- If you use dataset(s) or models from other authors, cite them appropriately in your work and follow their licenses.

Notes & next steps
------------------
- Replace placeholders (demo GIF, example outputs) with real media from your results folder.
- If you want, I can tailor this README with specific commands and file paths that match the code structure in your repo — share the layout or tell me where scripts/notebooks are and I’ll update the usage section accordingly.
