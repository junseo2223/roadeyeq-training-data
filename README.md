# RoadEyeQ Training Data

## Introduction
This repository contains training datasets and preprocessing resources for RoadEyeQ’s road-hazard detection models.
It supports the development of AI models that identify road hazards, including potholes, road cracks, and flooded road surfaces, using vehicle image and video data.

## Key Features
* Organizes image, video, and annotation data for road-hazard detection.
* Includes data preprocessing resources for AI model training.
* Supports structured dataset management for future model development and evaluation.
* Provides metadata resources, including location and road-condition information.

## Usage
1. Download or clone this repository.

git clone https://github.com/<your-username>/roadeyeq-training-data.git
cd roadeyeq-training-data

2. Add raw dataset files to the designated data directory.

3. Run the preprocessing scripts to prepare images, labels, and metadata for model training.

python preprocess.py

## License
This project is licensed under the MIT License.

## Dataset Preparation

`preprocess_dataset.py` validates YOLO-format annotations and prepares RoadEyeQ road-hazard data for model training. It creates deterministic train, validation, and test splits; generates an Ultralytics-compatible `data.yaml`; and saves a manifest and validation report.

### Expected source structure

```text
raw/
├── images/
│   ├── road_001.jpg
│   └── district_a/
│       └── road_002.jpg
└── labels/
    ├── road_001.txt
    └── district_a/
        └── road_002.txt
```

Each `.txt` file must use standard YOLO format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Run

```bash
python preprocess_dataset.py \
  --images raw/images \
  --labels raw/labels \
  --output-dir processed \
  --classes pothole,road_crack,flooded_surface
```

Use `--include-unlabeled` only for verified road images that contain no hazards; it will create empty labels for those negative samples.
