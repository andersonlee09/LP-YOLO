### 🌱 Overview

LP-YOLO is a lightweight object detection framework designed for maize ear endpoint localization in complex field environments.
The model focuses on improving the representation of weak structural keypoints (ear tip and ear base) under challenges such as:

Leaf occlusion

Illumination variation

Complex background interference

Small object sensitivity

To achieve this, LP-YOLO introduces a Progressive Feature Refinement Bottleneck (PFRB) and integrates it into a YOLO-based backbone, replacing conventional convolutional bottlenecks to enhance feature discrimination while maintaining computational efficiency.

### ⚙️ EnvironmentInstallation

1️⃣ Create environment
```
conda create -n lp_yolo python=3.9 -y
conda activate lp_yolo
```
2️⃣ Install dependencies
```
pip install ultralytics
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy pandas matplotlib
```

Dataset should follow YOLO format:
```
datasets/maize/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```
### 🏋️ Training

Run training script
```
python ultralytics/ear/train_ear.py
```