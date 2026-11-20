# Repository File Mapping & Migration Documentation

This document records the provenance, mapping, and sanitization of every file in the clean release repository (`~/reu_sie_release`) sourced from the research workspace (`<source_workspace>`).

---

## 1. Unavoidable File Renames

To maintain directory cleanliness and avoid filename collisions when placing both baseline and proposed models within the dataset directory:

| Original Source Path | Release Path | Reason for Rename |
| :--- | :--- | :--- |
| `SIE_VeRi/main.py` | `VeRi-776/train_sie_baseline.py` | Avoids collision with proposed `VeRi-776/main.py` (which trains `SIEQuatEncoder`). |

All other original script names have been strictly preserved.

---

## 2. Complete File Provenance & Mapping

### A. Root Configuration & Metadata Files

| Release Path | Source Path / Action | Description | Sanitization Applied |
| :--- | :--- | :--- | :--- |
| `README.md` | New | Skeleton README with installation, directory structure, and reproduction commands. | No results tables or metrics included (skeleton only). |
| `LICENSE` | Official GNU GPL-3.0 | GNU General Public License v3.0 text. | N/A |
| `NOTICE` | New | Copyright & Attribution Notice (acknowledging Meta Platforms, Inc. for original SIE base). | N/A |
| `requirements.txt` | Generated from `sie_new_environment.yml` | Pip requirements for installing project dependencies. | Clean package names without local CUDA build tags. |
| `environment.yml` | `sie_new_environment.yml` | Conda environment specification. | Removed `prefix: <user_home>/...` and all local username paths. |
| `.gitignore` | New | Git ignore patterns for checkpoints, weights, datasets, caches, and logs. | N/A |

---

### B. 3DIEBench (Continuous 3D Rotation, $SO(3)$)

Source directory: `SIE_STL/SIE_STL_version3_quat/`

| Release Path | Source Path | Description | Sanitization Applied |
| :--- | :--- | :--- | :--- |
| `3DIEBench/main.py` | `SIE_STL/SIE_STL_version3_quat/main.py` | Proposed method training script (`SIE_STL` with learned quaternion encoder). | Preserved Meta copyright header; dataset root defaults to `./data/3DIEBench`. |
| `3DIEBench/eval_classification.py` | `SIE_STL/SIE_STL_version3_quat/eval_classification.py` | Linear probe classification evaluation on invariant representations. | Preserved Meta header; paths made configurable via CLI. |
| `3DIEBench/eval_angle_prediction.py` | `SIE_STL/SIE_STL_version3_quat/eval_angle_prediction.py` | Rotation angle ($R^2$ / MSE) evaluation on equivariant representations. | Preserved Meta header; paths made configurable via CLI. |
| `3DIEBench/eval_feature_metrics.py` | `SIE_STL/SIE_STL_version3_quat/eval_feature_metrics.py` | Predictor cosine similarity and $L_2$ distance evaluation. | Added `--weights-file` CLI parameter; removed hardcoded checkpoint fallback. |
| `3DIEBench/eval_transformation_representation.py` | `SIE_STL/SIE_STL_version3_quat/eval_transformation_representation.py` | Transformation encoder representation evaluation. | Preserved Meta header; paths made configurable via CLI. |
| `3DIEBench/eval_prediction_errors.py` | `SIE_STL/SIE_STL_version3_quat/eval_prediction_errors.py` | Predictor error distribution evaluation. | Added `--weights-file` CLI parameter; removed hardcoded fallback. |
| `3DIEBench/eval_MRR_Hk.py` | `SIE_STL/SIE_STL_version3_quat/eval_MRR_Hk.py` | Equivariant retrieval evaluation (MRR and Hit@k). | Added `--weights-file` CLI parameter; removed hardcoded fallback. |
| `3DIEBench/data/main.py` | `SIE_STL/SIE_STL_version3_quat/data/main.py` | BlenderProc dataset rendering and view-pair sampling script. | Preserved Meta copyright header. |
| `3DIEBench/data/__init__.py` | New | Python package marker. | N/A |
| `3DIEBench/src/__init__.py` | New | Python package marker. | N/A |
| `3DIEBench/src/dataset.py` | `SIE_STL/SIE_STL_version3_quat/src/dataset.py` | 3DIEBench dataset loader and image array processor. | Preserved Meta header. |
| `3DIEBench/src/hypernetwork.py` | `SIE_STL/SIE_STL_version3_quat/src/hypernetwork.py` | Hypernetwork weight generator for the predictor. | Preserved Meta header. |
| `3DIEBench/src/models.py` | `SIE_STL/SIE_STL_version3_quat/src/models.py` | Model architectures (`SIE`, `SIE_STL`, projectors, gather layers). | Preserved Meta header. |
| `3DIEBench/src/predictors.py` | `SIE_STL/SIE_STL_version3_quat/src/predictors.py` | Equivariant predictor modules. | Preserved Meta header. |
| `3DIEBench/src/quaternion_transform.py` | `SIE_STL/SIE_STL_version3_quat/src/quaternion_transform.py` | `QuaternionEncoder`, geodesic loss, and quaternion math. | N/A |
| `3DIEBench/src/resnet.py` | `SIE_STL/SIE_STL_version3_quat/src/resnet.py` | ResNet-18 backbone architecture. | Preserved Meta header. |
| `3DIEBench/src/transformation_encoder.py` | `SIE_STL/SIE_STL_version3_quat/src/transformation_encoder.py` | Transformation encoder MLP backbone definitions. | Preserved Meta header. |
| `3DIEBench/src/transformation_pairing.py` | `SIE_STL/SIE_STL_version3_quat/src/transformation_pairing.py` | Pair construction and transformation targets. | Preserved Meta header. |

---

### C. VeRi-776 (Vehicle Viewpoint Transformation, $SO(2)$)

Source directories: `SIE_VeRi_V3Quat/` (proposed) and `SIE_VeRi/` (baseline)

| Release Path | Source Path | Description | Sanitization Applied |
| :--- | :--- | :--- | :--- |
| `VeRi-776/main.py` | `SIE_VeRi_V3Quat/main.py` | Proposed method training script (`SIEQuatEncoder`, $\lambda_{\text{trans}}=60, \lambda_{\text{std}}=35, \lambda_{\text{sim}}=5$). | Preserved Meta header; defaults to relative data paths. |
| `VeRi-776/train_sie_baseline.py` | `SIE_VeRi/main.py` | True-label baseline training script (original SIE with 2000 epochs). | Preserved Meta header; renamed to prevent collision with proposed `main.py`. |
| `VeRi-776/eval_classification.py` | `SIE_VeRi_V3Quat/eval_classification.py` | Linear probe classification evaluation on invariant representation. | Preserved Meta header; paths configurable via CLI. |
| `VeRi-776/eval_angle_prediction.py` | `SIE_VeRi_V3Quat/eval_angle_prediction.py` | Rotation angle ($R^2$ / MSE) evaluation on equivariant representations. | Evaluates both proposed and baseline models via `--weights-file`. |
| `VeRi-776/eval_predictor_metrics.py` | `SIE_VeRi_V3Quat/eval_predictor_metrics.py` | Predictor cosine similarity and $L_2$ distance evaluation. | Paths configurable via CLI. |
| `VeRi-776/eval_map_cmc.py` | `SIE_VeRi_V3Quat/eval_map_cmc.py` | Vehicle Re-ID retrieval evaluation (mAP and CMC Rank-1/5) for proposed model. | Paths configurable via CLI. |
| `VeRi-776/eval_map_cmc_old_sie.py` | `SIE_VeRi/eval_map_cmc_old_sie.py` | Vehicle Re-ID retrieval evaluation for the true-label SIE baseline. | Removed hardcoded `<source_workspace>` paths; added CLI arguments. |
| `VeRi-776/eval_MRR_Hk.py` | `SIE_VeRi_V3Quat/eval_MRR_Hk.py` | Equivariant retrieval evaluation (MRR and Hit@k). | Added `--weights-file` CLI argument; removed hardcoded fallback. |
| `VeRi-776/eval_prediction_errors.py` | `SIE_VeRi_V3Quat/eval_prediction_errors.py` | Predictor error distribution evaluation. | Added `--weights-file` CLI argument; removed hardcoded fallback. |
| `VeRi-776/analysis/analyze_orientation.py` | `SIE_VeRi_V3Quat/analysis/analyze_orientation.py` | Orientation distribution analysis script. | Cleaned relative data paths. |
| `VeRi-776/analysis/analyze_veri.py` | `SIE_VeRi_V3Quat/analysis/analyze_veri.py` | VeRi XML metadata and attribute parser. | Cleaned relative data paths. |
| `VeRi-776/analysis/verify_dataset.py` | `SIE_VeRi_V3Quat/analysis/verify_dataset.py` | Dataset integrity verification script. | Cleaned relative data paths. |
| `VeRi-776/analysis/visualize_keypoints.py`| `SIE_VeRi_V3Quat/analysis/visualize_keypoints.py` | Vehicle keypoint and bounding box visualizer. | Cleaned relative data paths. |
| `VeRi-776/analysis/__init__.py` | New | Python package marker. | N/A |
| `VeRi-776/src/__init__.py` | New | Python package marker. | N/A |
| `VeRi-776/src/dataset.py` | `SIE_VeRi_V3Quat/src/dataset.py` | Dynamic on-the-fly vehicle pair dataset loader. | Cleaned relative data paths. |
| `VeRi-776/src/dataset_veri.py` | `SIE_VeRi_V3Quat/prototype/dataset_veri.py` | Standalone VeRi dataset loader implementation. | Cleaned hardcoded dataset paths. |
| `VeRi-776/src/hypernetwork.py` | `SIE_VeRi_V3Quat/src/hypernetwork.py` | Hypernetwork weight generator for VeRi. | Preserved Meta header. |
| `VeRi-776/src/models.py` | `SIE_VeRi_V3Quat/src/models.py` | Model architectures (`SIE`, `SIEQuatEncoder`, evaluators). | Preserved Meta header. |
| `VeRi-776/src/predictors.py` | `SIE_VeRi_V3Quat/src/predictors.py` | Predictor heads. | Preserved Meta header. |
| `VeRi-776/src/quaternion_transform.py` | `SIE_VeRi_V3Quat/src/quaternion_transform.py` | `QuaternionEncoder` and geodesic losses. | N/A |
| `VeRi-776/src/resnet.py` | `SIE_VeRi_V3Quat/src/resnet.py` | ResNet-18 feature extractor. | Preserved Meta header. |

---

### D. DN-348 (Categorical Day/Night Domain Shift, $\mathbb{Z}_4$)

Source directory: `DN348_SIE/`

| Release Path | Source Path | Description | Sanitization Applied |
| :--- | :--- | :--- | :--- |
| `DN-348/main.py` | `DN348_SIE/main.py` | Training entrypoint for `SIECategoricalOracle` and unconditioned SIE. | Preserved Meta header; defaults to relative data paths. |
| `DN-348/eval_predictor_metrics_dn348.py` | `DN348_SIE/eval_predictor_metrics_dn348.py` | Predictor metrics on DN-348 pair splits. | Cleaned dataset default paths. |
| `DN-348/eval_map_cmc_dn348.py` | `DN348_SIE/eval_map_cmc_dn348.py` | Cross-domain vehicle Re-ID retrieval evaluation (Day $\leftrightarrow$ Night). | Cleaned dataset default paths. |
| `DN-348/eval_classification_dn348.py` | `DN348_SIE/eval_classification_dn348.py` | Classification linear probe evaluator. | Cleaned dataset default paths. |
| `DN-348/scripts/train_sie_identity_supcon_5k.py` | `DN348_SIE/scripts/train_sie_identity_supcon_5k.py` | Proposed Identity SupCon training stage 1 (Steps 0–5000). | Removed hardcoded sys.path inserts and `<user_home>` defaults; uses relative package paths. |
| `DN-348/scripts/train_sie_identity_supcon_10k.py` | `DN348_SIE/scripts/train_sie_identity_supcon_10k.py` | Proposed Identity SupCon training stage 2 (Steps 5000–10000; yields Step 9000). | Removed hardcoded sys.path inserts and `<user_home>` defaults; uses relative package paths. |
| `DN-348/scripts/eval_invariant_classification.py`| `DN348_SIE/scripts/eval_invariant_classification.py` | Linear probe evaluation for invariant representation on 200 identities. | Removed `<user_home>` defaults. |
| `DN-348/scripts/evaluate_transformation_encoder.py`| `DN348_SIE/scripts/evaluate_transformation_encoder.py` | Transformation encoder accuracy, F1, Cosine, and $L_2$ evaluation. | Removed `<user_home>` defaults. |
| `DN-348/scripts/eval_predictor_metrics_oracle.py`| `DN348_SIE/scripts/eval_predictor_metrics_oracle.py` | Predictor metrics for the `SIECategoricalOracle` baseline. | Removed `<user_home>` defaults. |
| `DN-348/scripts/eval_master_transformation_encoder.py`| `DN348_SIE/scripts/eval_master_transformation_encoder.py` | Unified cross-dataset transformation encoder evaluation (Table 2). | Updated repository roots to reference `3DIEBench`, `VeRi-776`, `DN-348` in the release tree. |
| `DN-348/scripts/train_resnet18_baseline.py` | `DN348_SIE/scripts/train_resnet18_baseline.py` | ResNet-18 standard supervised cross-entropy baseline training. | Removed `<user_home>` defaults. |
| `DN-348/scripts/eval_classification_resnet18_baseline.py` | `DN348_SIE/scripts/eval_classification_resnet18_baseline.py` | Linear probe classification evaluation for ResNet-18 baseline. | Removed `<user_home>` defaults. |
| `DN-348/scripts/eval_diagnostics_resnet18_baseline.py` | `DN348_SIE/scripts/eval_diagnostics_resnet18_baseline.py` | Representation rank and singular value diagnostics for ResNet-18 baseline. | Removed `<user_home>` defaults. |
| `DN-348/scripts/build_train_pairs.py` | `DN348_SIE/scripts/build_train_pairs.py` | DN-348 pairwise dataset generator for training. | Cleaned default paths to relative `./data/dn348`. |
| `DN-348/scripts/build_train_val_pairs.py` | `DN348_SIE/scripts/build_train_val_pairs.py` | DN-348 balanced train/val pair generator (3,200 validation pairs). | Cleaned default paths to relative `./data/dn348`. |
| `DN-348/scripts/verify_pair_labels.py` | `DN348_SIE/scripts/verify_pair_labels.py` | Integrity verifier for generated pair files. | Cleaned default paths. |
| `DN-348/scripts/__init__.py` | New | Python package marker. | N/A |
| `DN-348/src/__init__.py` | New | Python package marker. | N/A |
| `DN-348/src/backbone.py` | `DN348_SIE/src/backbone.py` | ResNet backbone wrapper. | **Fixed cross-repo import**: Replaced `sys.path.insert(0, .../SIE_VeRi_V3Quat/src)` with `from . import resnet`. |
| `DN-348/src/dataset.py` | `DN348_SIE/src/dataset.py` | Generic DN-348 image pair loader. | Preserved Meta header. |
| `DN-348/src/dataset_dn348.py` | `DN348_SIE/src/dataset_dn348.py` | DN-348 official split image loader. | Preserved Meta header. |
| `DN-348/src/dataset_dn348_supcon.py` | `DN348_SIE/src/dataset_dn348_supcon.py` | Identity-balanced batch sampler and pair generator. | N/A |
| `DN-348/src/daynight_transform.py` | `DN348_SIE/src/daynight_transform.py` | Day/Night discrete transformation encoder, losses, and inverse consistency. | N/A |
| `DN-348/src/dn_transform.py` | `DN348_SIE/src/dn_transform.py` | Discrete transition classification loss utilities. | N/A |
| `DN-348/src/hypernetwork.py` | `DN348_SIE/src/hypernetwork.py` | Hypernetwork weight generator. | Preserved Meta header. |
| `DN-348/src/models.py` | `DN348_SIE/src/models.py` | Model architectures (`SIECategoricalOracle`, `SIEDayNightEncoder`). | Preserved Meta header. |
| `DN-348/src/predictors.py` | `DN348_SIE/src/predictors.py` | Predictor networks. | Preserved Meta header. |
| `DN-348/src/quaternion_transform.py` | `DN348_SIE/src/quaternion_transform.py` | Quaternion math utilities. | N/A |
| `DN-348/src/resnet.py` | `DN348_SIE/src/resnet.py` | ResNet-18 architecture. | Preserved Meta header. |
| `DN-348/src/resnet18_baseline.py` | `DN348_SIE/src/resnet18_baseline.py` | Supervised baseline ResNet-18 architecture. | N/A |
| `DN-348/src/supcon_loss.py` | `DN348_SIE/src/supcon_loss.py` | Supervised Contrastive Loss implementation. | N/A |

---

### E. Configuration Files (`configs/`)

Dumped hyperparameter configurations from the validated training runs:

| Release Path | Source Path | Description |
| :--- | :--- | :--- |
| `configs/3die_params_proposed_500ep.json` | `SIE_STL/SIE_STL_version3_quat/checkpoints/v3_quat_200ep_to_500ep_20261005_144431/params.json` | 3DIEBench proposed method hyperparameters (500 epochs). |
| `configs/veri776_params_baseline_2000ep.json` | `SIE_VeRi/checkpoints/veri_2000ep/params.json` | VeRi-776 true-label baseline hyperparameters (2000 epochs). |
| `configs/veri776_params_proposed_700ep.json` | `SIE_VeRi_V3Quat/checkpoints/veri_quatencoder_tw60_50ep_20260822_152234/params.json` | VeRi-776 proposed method hyperparameters (700 epochs). |
| `configs/dn348_params_oracle_baseline.json` | `DN348_SIE/checkpoints/dn348_sie_oracle/params.json` | DN-348 Oracle baseline hyperparameters (200 epochs). |
| `configs/dn348_params_proposed_step9000.json` | `DN348_SIE/checkpoints/dn348_sie_identity_supcon/params.json` | DN-348 Identity-Balanced SupCon proposed hyperparameters (Step 9000). |
| `configs/dn348_params_resnet18_baseline.json` | `DN348_SIE/checkpoints/dn348_resnet18_identity_baseline/params.json` | DN-348 ResNet-18 supervised baseline hyperparameters. |
