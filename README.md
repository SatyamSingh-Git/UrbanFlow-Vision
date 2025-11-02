<div align="center">

# UrbanFlow Vision

Real‑time traffic intelligence from street cameras: detect, track, project to road‑plane, estimate speed, and export to OpenSCENARIO.

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](./environment.yml)
![OS](https://img.shields.io/badge/OS-Windows%2010%2F11%20%7C%20Linux-success)
![GPU](https://img.shields.io/badge/GPU-Optional-black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](./Final/LICENSE)

</div>

---

## Overview

UrbanFlow Vision turns raw traffic camera footage into actionable insights. The pipeline combines modern object detection (Detectron2), multi‑object tracking (DeepSORT), and single‑view geometry to project image coordinates onto the road plane. From there, it computes instantaneous and average speeds, visualizes trajectories, and exports scenarios to OpenSCENARIO for simulation.

What you get in minutes:

- Vehicle and pedestrian detection/tracking
- Metric trajectories on the ground plane (meters)
- Speed estimation (m/s and km/h)
- Plots and OpenSCENARIO (.xosc) export
- Optional video harvesting from TfL cameras

## Key features

- Modern detector (Detectron2) with instance masks
- Robust multi‑object tracking with DeepSORT
- Automatic scale estimation from early frames
- Road‑plane projection and trajectory building
- Speed analytics (instantaneous + median average)
- One‑click exports: plots and OpenSCENARIO
- Works CPU‑only; GPU accelerates everything

## Project structure

- `Final/` — all runnable code and assets live here
	- `Overall.py` — end‑to‑end pipeline (detect → track → project → write coords)
	- `get_videos.py` — optional TfL video downloader
	- `get_trajectory.py` — create trajectory plots from coords
	- `get_speed.py` — compute speeds from coords + video FPS
	- `ToSimulation.py` — export OpenSCENARIO (.xosc)
	- `detectron2_detection.py`, `deep_sort/` — core vision modules
	- `cfg/`, `data/` — model configs and label files
- `environment.yml` — base Conda environment (CPU PyTorch by default)
- `Final/LICENSE` — GPL‑3.0 license

## Quick start (Windows and Linux)

Prerequisites:

- Conda (Anaconda/Miniconda)
- Python 3.8 (environment.yml creates it for you)
- Optional: NVIDIA GPU with recent CUDA + drivers
- Detectron2 installed inside the Conda env (follow the official guide for your OS/GPU)

### 1) Create the Conda environment

```cmd
conda env create -f environment.yml
conda activate app
```

Note: The provided env is CPU‑only. If you have a GPU, install a CUDA build of PyTorch and then Detectron2 appropriate for your CUDA version.

### 2) Install Detectron2

Follow the official instructions for your platform and CUDA: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

Tip (Windows): ensure you have the Microsoft C++ Build Tools installed to satisfy native extensions when needed.

### 3) Verify imports

```cmd
python -c "import torch, cv2; print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

## Run the pipeline

All commands below are executed from the `Final` folder.

```cmd
cd Final
```

- Optional: grab sample traffic clips (TfL public cameras)

```cmd
python get_videos.py
```

- Run detection → tracking → projection and write coordinates to file

```cmd
python Overall.py video_sample1.mp4 --coordfile coords.txt --use_cuda False
```

For GPU (if your env has CUDA‑enabled PyTorch and Detectron2):

```cmd
python Overall.py video_sample1.mp4 --coordfile coords.txt --use_cuda True
```

- Create an output folder for analytics

```cmd
mkdir Results
```

- Plot trajectories

```cmd
python get_trajectory.py --input_coords coords.txt --output_name demo
```

- Estimate speeds (prints instantaneous and average; writes to Results)

```cmd
python get_speed.py --input_video video_sample1.mp4 --input_coords coords.txt --output_speeds demo_speed.txt
```

- Export OpenSCENARIO (.xosc)

```cmd
python ToSimulation.py --input_video video_sample1.mp4 --input_coords coords.txt --output_name demo
```

Outputs:

- `Results/demo.png` — trajectory plot
- `Results/demo_speed.txt` — speed report
- `Results/demo.xosc` — OpenSCENARIO export

## Configuration tips

- Classes and labels live in `Final/data/*.names` (e.g., `coco.names`).
- Detector/tracker behavior is configured in `Final/detectron2_detection.py` and `Final/deep_sort/`.
- Model configs in `Final/cfg/`; adjust to experiment with different YOLO or other configs if you extend the repo.

## Troubleshooting

- Detectron2 on Windows: follow the official guide carefully; match PyTorch, CUDA and Visual Studio toolchain versions.
- CPU‑only is supported, just slower. Set `--use_cuda False`.
- If trajectory or speed scripts error with a missing folder, run `mkdir Results` first.
- If you see empty detections, confirm Detectron2 weights are accessible and the classes file matches your model.

## Demo video

- Walkthrough: https://www.youtube.com/watch?v=nomez0_uHzo

## Ethics and use

Please use responsibly and comply with local laws and privacy regulations. This project is intended for research, safety, and simulation—not for intrusive surveillance.

## License

GPL‑3.0 — see `Final/LICENSE`.

## Credits

This project builds on prior academic and open‑source work in detection, tracking, and simulation. Original foundations and inspiration by contributors including:

- Spyridon Couvaras
- Prof. Yiannis Demiris and the Personal Robotics Lab
- Transport Systems and Logistics Laboratory (TSL)

If your work is used here and you’d like an explicit citation or update, please open an issue.
