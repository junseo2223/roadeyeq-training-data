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
