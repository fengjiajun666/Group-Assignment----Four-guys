#  YOLOv8 Car Detection & Classification (Stanford Cars)

**Course:** Machine Learning 
**Group 13:** Four Guys

##  Project Overview
This project implements an end-to-end machine learning application using the YOLOv8 architecture to detect and classify 196 fine-grained car models. The pipeline includes data parsing, exploratory data analysis (EDA), deep learning model training, and a user-friendly PyQt5 graphical interface for real-time inference.

##  Dataset
* **Source:** [Stanford Cars Dataset](https://github.com/jhpohovey/StanfordCars-Dataset)
* **Details:** 16,185 images across 196 car classes (e.g., Make, Model, Year).
* **Task Type:** Object Detection & Fine-grained Image Classification.

##  Setup & Environment Configuration
Ensure you have Python 3.8+ installed. All required dependencies (including `ultralytics`, `scipy` for `.mat` parsing, `opencv-python`, and `PyQt5`) are listed in the requirements file.

##  Step-by-Step Execution Guide

### Step 1: Data Preprocessing & Formatting
The original dataset uses MATLAB (`.mat`) annotations. We provide a script to parse these files and convert the absolute bounding box coordinates into normalized YOLO `.txt` format.
```bash
# Run the conversion script (ensure images are placed in datasets/StanfordCars)
python detect_tools.py
```

### Step 2: Exploratory Data Analysis (EDA)
To view the underlying data distributions (class balance, bounding box sizes, etc.), run our EDA script. This will generate analytical plots saved directly in the root directory.
```bash
python eda.py
```

### Step 3: Model Training
To train the YOLOv8 model from scratch on the parsed dataset, execute the training script. The script automatically uses `stanford_cars.yaml` for configuration.
```bash
python train.py
```
*(Note: Training outputs, weights, and evaluation curves will be saved in the `runs/detect/train/` directory. Our best model achieved a mAP@0.5 of 0.949).*


##  Contributors
* Project Manager & Report Lead: [FENG Jiajun]
* Data Engineer (EDA & Preprocessing): [HE Yudong]
* Algorithm Engineer (YOLO Pipeline): [Jia Zixin]
* Evaluation & Deployment (GUI & GitHub): [GUAN Beicheng]
