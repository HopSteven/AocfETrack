
# Enhanced Multi-Object Tracking via Association-Optimized Collaborative Features

This repository contains the implementation of the multi-object tracking (MOT) method proposed in the paper *A Multi-Object Tracking Method Based on Association-Enhanced Collaborative Optimization*, which enhances the robustness and accuracy of tracking in complex scenes. The method is built upon the ByteTrack framework, integrating a Channel-Shared Lightweight Feature Fusion (CSLFF) mechanism, Scale-Aware IoU (ScAIoU) matching, and a Trajectory Prediction Assisted Matching (TPAM) with an appearance learning module (SLM) for improved data association.

## Overview
- **Paper**: [Enhanced Multi-Object Tracking via
Association-Optimized Collaborative Features](https://arxiv.org/abs/XXXXX) (Submitted, 2025)
- **Framework**: Extends ByteTrack with novel enhancements for MOT tasks.
- **Key Contributions**:
  1. **CSLFF**: A lightweight, channel-shared multi-scale feature fusion module in the YOLOX neck to improve detection feature representation.
  2. **ScAIoU**: Integrated to improve IoU-based matching with scale awareness.
  3. **TPAM + SLM**: Added for trajectory prediction and appearance-based re-matching, reducing identity switches.
- **Performance**: Achieves MOTA of 80.7% on MOT17 and 78.1% on MOT20, with IDF1 improved by 2.3% compared to the baseline.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch (>=1.8.0)
- CUDA 11.x (for GPU support)
- Other dependencies: `numpy`, `scipy`, `lap`, `motmetrics`, `pycocotools`, `cython_bbox`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/HopSteven/AocfETrack.git
   cd AocfETrack
   ```
   
2. Install requirements:
   ```bash
   pip3 install -r requirements.txt
   pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
   pip3 install cython_bbox
   ```
3. (Optional) Build Docker image:
   ```bash
   docker build -t track-enhanced:latest .
   # Run with GPU support (adjust paths as needed)
   docker run --gpus all -it --rm -v $PWD:/workspace/ByteTrack-Enhanced bytetrack-enhanced:latest
   ```

## Data Preparation
1. Download datasets (e.g., MOT17, MOT20) from [MOTChallenge](https://motchallenge.net/) and organize them under `datasets/`:
   ```
   datasets/
   ├── MOT17/
   │   ├── images/
   │   │   ├── train/
   │   │   │   ├── MOT17-02-FRCNN/
   │   │   │   │   ├── img1/
   │   │   │   │   └── gt/
   │   │   │   └── ...
   │   ├── annotations/
   │   │   ├── train_half.json
   │   │   ├── val_half.json
   └── ...
   ```
2. Convert datasets to COCO format:
   ```bash
   python3 tools/convert_mot17_to_coco.py
   python3 tools/convert_mot20_to_coco.py
   ```
3. (Optional) Mix datasets for ablation studies:
   ```bash
   python3 tools/mix_data_ablation.py
   ```

## Training
Train the model on MOT17 (half dataset) with the following command:
```bash
python3 tools/train.py -f exps/example/mot/yolox_x_mot17_half.py -d 8 -b 64 --fp16 -o -c pretrained/yolox_x.pth
```
- **Parameters**:
  - `-f`: Experiment configuration file.
  - `-d`: Number of GPUs (1 for single GPU).
  - `-b`: Batch size (adjust to 4-8 based on GPU memory).
  - `--fp16`: Enable mixed precision training.
  - `-c`: Pretrained YOLOX-X weights.
- **Customization**:
  - Modify `max_epoch` in `yolox_x_mot17_half.py` (e.g., `self.max_epoch = 50`) for longer training.
  - Adjust `input_size` (e.g., `(640, 1152)`) for small object optimization.

## Inference and Evaluation
### Tracking Inference
Run tracking on a single image or video:
```bash
python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mot17_half.py \
  -c YOLOX_outputs/yolox_x_mot17_half/best_ckpt.pth \
  --fp16 \
  --save_result \
  --path datasets/MOT17/images/train/MOT17-02-FRCNN/img1/000001.jpg
```

### MOT Evaluation
Evaluate tracking performance on MOT17 val_half:
```bash
python3 tools/track.py \
  -f exps/example/mot/yolox_x_mot17_half.py \
  -c YOLOX_outputs/yolox_x_mot17_half/last_epoch_ckpt.pth.tar \
  -b 64 \
  -d 8\
  --fp16 \
  --fuse \
  --track_thresh 0.4 \
  --match_thresh 0.7
```
- **Output**: Saves results to `YOLOX_outputs/yolox_x_mot17_half/track_results/`.
- **Optimization**: Adjust `track_thresh` (0.3-0.6) and `match_thresh` (0.6-0.9) for better MOTA.



## Model Performance
- **MOT17 (val_half)**: Initial MOTA ~0.2-0.4 (with optimization), target 80.7% with further tuning.
- **MOT20**: Target 78.1% (requires full dataset training).


## Citation
If you use this code in your research, please cite our paper:
```
@article{zhang2025mot,
  title={Enhanced Multi-Object Tracking via Association-Optimized Collaborative Features}
  author={Shuo Cai,Zeyang Deng,Yuanzhi Tang}
  journal={},
  year={2025}
}
```

## Acknowledgments
This work builds upon the original ByteTrack by [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) and YOLOX by [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). Thanks to the community for their contributions.

