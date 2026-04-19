QMViT Project - Full Fixed Scaffolding (zones removed)

This package contains:
 - data/train.csv support: qmvit_train will use data/train.csv if present (expects columns: filepath, stage, plus)
 - models/swin_unet.py    : a lightweight Swin-style UNet implemented in PyTorch
 - models/qmvit.py       : QMViT model that uses PennyLane if available, otherwise a classical fallback
 - scripts/segmentation_train.py : training stub for segmentation using TinySwinUNet
 - scripts/qmvit_train.py : training stub for QMViT classification training
 - scaffold.py           : orchestration script to run segmentation and QMViT training
 - requirements.txt      : suggested packages to install (see below)

Important notes:

 - The QMViT quantum preprocessing uses PennyLane's default.qubit device if available. If PennyLane or
   a valid device is not available, QMViT falls back to a small classical module that simulates the effect
   of a quanvolution layer so you can still run experiments and compare.

Data layout expected (zone-free):
  data/
    stages/
      stage1/  (images)
      stage2/
      stage3/
      stage4/
      normal/
    plus/
      plus/
      preplus/
      normal/
    segmentation/      (optional; used if --do_seg)
      images/
      masks/

Quick start (after installing dependencies):
  python scaffold.py --data_dir ./data --output_dir ./outputs --do_seg --do_qmvit --task stages --epochs 5

