# Equivariant-Invariant Representation Learning with Transformation-Conditioned Hypernetworks

This repository provides the official implementation of our equivariant-invariant representation learning framework conditioning Hypernetworks via learned transformation encoders across 3D object rotation, vehicle viewpoint variation, and day/night domain shifts.

---

## 1. Repository Structure

```
.
├── 3DIEBench/           # 3D continuous rotation experiments (SO(3))
│   ├── main.py          # Proposed training script (quaternion encoder)
│   ├── eval_*.py        # Evaluation scripts (classification, R2, predictor)
│   ├── data/            # Rendering & sampling utilities
│   └── src/             # Model components & quaternion math
├── VeRi-776/            # Vehicle viewpoint transformation experiments (SO(2))
│   ├── main.py          # Proposed training script (SIEQuatEncoder)
│   ├── train_sie_baseline.py # Original SIE true-label baseline training
│   ├── eval_*.py        # Evaluation scripts (mAP/CMC, R2, predictor)
│   ├── analysis/        # Dataset orientation & keypoint inspection
│   └── src/             # Model architectures & dataset loader
├── DN-348/              # Day/night domain transition experiments (Z_4)
│   ├── main.py          # Oracle baseline training entrypoint
│   ├── scripts/         # Staged training (5k, 10k), baselines & evaluations
│   ├── eval_*.py        # Retrieval and predictor metric evaluators
│   └── src/             # Identity SupCon loss, backbone & transformation encoder
├── configs/             # Reference hyperparameter parameters from final runs
├── docs/                # Migration file map and documentation
├── environment.yml      # Conda environment definition
├── requirements.txt     # Python package requirements
├── LICENSE              # GNU General Public License v3.0 (GPL-3.0)
└── NOTICE               # Attribution & copyright notices
```

---

## 2. Environment Setup

### Using Conda
```bash
conda env create -f environment.yml
conda activate sie_new
```

### Using Pip
```bash
pip install -r requirements.txt
```

---

## 3. Pretrained Model Weights

Inference-only checkpoints (containing model weights without optimizer states) can be placed in `./weights/`:
- `3die_epoch500.pth` (3DIEBench proposed model, 500 epochs)
- `veri776_epoch700.pth` (VeRi-776 proposed model, 700 epochs)
- `dn348_step9000.pth` (DN-348 proposed Identity SupCon model, Step 9000)

---

## 4. Training

### 3DIEBench
- **True-Label Baseline**: Refer to the official [Meta SIE repository](https://github.com/facebookresearch/sie).
- **Proposed Method**:
  ```bash
  python 3DIEBench/main.py \
    --experience SIE_STL \
    --dataset-root ./data/3DIEBench \
    --images-file ./data/3DIEBench/train_images.npy \
    --labels-file ./data/3DIEBench/train_labels.npy \
    --exp-dir ./checkpoints/3die_proposed \
    --epochs 500 --batch-size 64 --base-lr 0.001
  ```

### VeRi-776
- **True-Label Baseline**:
  ```bash
  python VeRi-776/train_sie_baseline.py \
    --experience SIE \
    --dataset-root ./data/VeRi \
    --exp-dir ./checkpoints/veri_baseline \
    --epochs 2000 --batch-size 8 --base-lr 0.001
  ```
- **Proposed Method**:
  ```bash
  python VeRi-776/main.py \
    --experience SIEQuatEncoder \
    --dataset-root ./data/VeRi \
    --exp-dir ./checkpoints/veri_proposed \
    --epochs 700 --batch-size 64 --base-lr 0.001 \
    --trans-weight 60.0 --std-coeff 35.0 --sim-coeff 5.0
  ```

### DN-348
- **Oracle Baseline**:
  ```bash
  python DN-348/main.py \
    --experience SIECategoricalOracle \
    --dataset-root ./data/dn348 \
    --exp-dir ./checkpoints/dn348_oracle \
    --epochs 200 --batch-size 64 --base-lr 0.001
  ```
- **Supervised ResNet-18 Baseline**:
  ```bash
  python DN-348/scripts/train_resnet18_baseline.py \
    --dataset-root ./data/dn348 \
    --exp-dir ./checkpoints/dn348_resnet18_baseline \
    --epochs 50 --batch-size 64 --lr 0.001
  ```
- **Proposed Method (Two-Stage Identity SupCon)**:
  ```bash
  # Stage 1: Steps 0 -> 5000
  python DN-348/scripts/train_sie_identity_supcon_5k.py \
    --dataset-root ./data/dn348 \
    --weights-file ./weights/veri776_epoch700.pth \
    --exp-dir ./checkpoints/dn348_identity_supcon \
    --target-steps 5000

  # Stage 2: Steps 5000 -> 10000 (checkpoint at Step 9000)
  python DN-348/scripts/train_sie_identity_supcon_10k.py \
    --dataset-root ./data/dn348 \
    --weights-file ./weights/veri776_epoch700.pth \
    --exp-dir ./checkpoints/dn348_identity_supcon \
    --target-steps 10000
  ```

---

## 5. Evaluation

### 3DIEBench
```bash
# Linear probe classification (Invariant representation)
python 3DIEBench/eval_classification.py \
  --dataset-root ./data/3DIEBench \
  --weights-file ./weights/3die_epoch500.pth \
  --inv-part --epochs 50 --batch-size 128

# Rotation angle R2 prediction (Equivariant representation)
python 3DIEBench/eval_angle_prediction.py \
  --dataset-root ./data/3DIEBench \
  --weights-file ./weights/3die_epoch500.pth \
  --deep-end --equi-dims 256 --epochs 50

# Predictor cosine similarity & L2 distance
python 3DIEBench/eval_feature_metrics.py \
  --dataset-root ./data/3DIEBench \
  --weights-file ./weights/3die_epoch500.pth

# Transformation encoder representation evaluation
python 3DIEBench/eval_transformation_representation.py \
  --dataset-root ./data/3DIEBench \
  --weights-file ./weights/3die_epoch500.pth
```

### VeRi-776
```bash
# Linear probe classification
python VeRi-776/eval_classification.py \
  --dataset-root ./data/VeRi \
  --weights-file ./weights/veri776_epoch700.pth \
  --num-classes 575 --inv-part --epochs 100

# Viewpoint angle R2 prediction
python VeRi-776/eval_angle_prediction.py \
  --dataset-root ./data/VeRi \
  --weights-file ./weights/veri776_epoch700.pth \
  --num-classes 575 --deep-end --equi-dims 512

# Predictor cosine similarity & L2 distance
python VeRi-776/eval_predictor_metrics.py \
  --dataset-root ./data/VeRi \
  --weights-file ./weights/veri776_epoch700.pth

# Vehicle Re-ID retrieval (mAP / CMC)
python VeRi-776/eval_map_cmc.py \
  --dataset-root ./data/VeRi \
  --weights-file ./weights/veri776_epoch700.pth

# Original SIE Baseline Re-ID retrieval
python VeRi-776/eval_map_cmc_old_sie.py \
  --dataset-root ./data/VeRi \
  --checkpoint-path ./weights/veri_baseline.pth
```

### DN-348
```bash
# Invariant linear probe classification
python DN-348/scripts/eval_invariant_classification.py \
  --checkpoint ./weights/dn348_step9000.pth \
  --dataset-root ./data/dn348 \
  --batch-size 64 --lr 0.01 --epochs 50

# Transformation prediction accuracy & representation metrics
python DN-348/scripts/evaluate_transformation_encoder.py \
  --checkpoint ./weights/dn348_step9000.pth \
  --dataset-root ./data/dn348

# Predictor metrics (conditioned on predicted transformation)
python DN-348/eval_predictor_metrics_dn348.py \
  --checkpoint ./weights/dn348_step9000.pth \
  --dataset-root ./data/dn348 \
  --pairs-file ./data/splits/val_pairs.txt

# Oracle baseline predictor metrics
python DN-348/scripts/eval_predictor_metrics_oracle.py \
  --checkpoint ./weights/dn348_oracle.pth \
  --dataset-root ./data/dn348 \
  --pairs-file ./data/splits/val_pairs.txt

# Cross-domain vehicle Re-ID retrieval (Day <-> Night)
python DN-348/eval_map_cmc_dn348.py \
  --weights-file ./weights/dn348_step9000.pth \
  --dataset-root ./data/dn348
```

---

## 6. License & Attribution

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) — see [LICENSE](LICENSE) for details.
Components derived from the original Meta SIE implementation are acknowledged in [NOTICE](NOTICE).
