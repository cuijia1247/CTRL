# StyleConsensusLearning

### Description

**TCFL (Triangular/Diamond-shape Consensus Feature Learning)** — an artistic style classification framework based on community feature learning. It uses pretrained CNN backbones (VGG16 / ResNet50) for feature extraction, and a novel **Style Learning AutoEncoder (SLAE)** with a configurable triangle or diamond network topology to learn style-consensus features across same-class images. A lightweight linear classifier is trained on top.

---

### Architecture

```
Image
  └─► CNN Backbone (VGG16 / ResNet50)          feature extraction
        └─► StyleLearningAutoEncoder (SLAE)     style consensus learning
              ├─ Triangle topology  (ae1→ae2→ae3)
              └─ Diamond topology   (DiamondStyleLearningCell, Gram-loss)
                    └─► Linear Classifier       top-1/top-3 prediction
```

**Key modules:**

| File | Description |
|---|---|
| `model_NEW.py` | SLAE, CDAutoEncoder, StyleLearningCell1/2/3, DiamondStyleLearningCell |
| `SCLdataSet_NEW.py` | `TCFLDataset` — loads features / images, computes per-sample style levels |
| `run_NEW.py` | Training loop (feature mode or image mode) |
| `run_test_NEW.py` | Evaluation, batch prediction, dataset image filtering |
| `style_predict.py` | Single-image / folder-level style prediction |
| `MlpNet.py` | Optional MLP classifier |
| `utils/CustomLoss.py` | KL, Cosine, Cityblock, Chebyshev, DotProduct, **Gram** loss |
| `utils/styleLevelCal.py` | Centroid-based Euclidean style-level assignment |
| `utils/trainTestSplit.py` | Train / test label file generation |
| `utils/trainTestValSplit.py` | Train / val / test label file generation |
| `utils/copy_file.py` | Copy images to class sub-directories |
| `utils/image_processing.py` | Image I/O helpers |

---

### Style Level Assignment

Each sample is assigned a **style level** (0–3) according to its Euclidean distance from the class centroid:

| Level | Distance range |
|---|---|
| 3 (strongest) | < 10 % of range from centroid |
| 2 | 10 %–30 % |
| 1 | 30 %–50 % |
| 0 (weakest) | ≥ 50 % |

The assigned style level determines which **same-class nearest-neighbour** feature is used as the consensus target for the SLAE cell during training.

---

### Supported Datasets

| Dataset | # Classes | Notes |
|---|---|---|
| Painting91 | 91 | Classical painting styles |
| Pandora | — | Diverse artistic styles |
| WikiArt3 | 15 | WikiArt subset |
| FashionStyle14 | 14 | Fashion photography |
| AVAStyle | — | Aesthetic visual analysis |
| Arch | 25 | Architectural styles |
| webstyle | 10 | Web-UI design styles |

---

### Installation

```bash
python==3.9.18
pytorch==1.12.0
torchvision==0.13.0
pillow==10.0.1
cudatoolkit==11.3.1
scipy==1.11.4          # pip install scipy
tqdm==4.66.1           # pip install tqdm
matplotlib==3.8.2      # pip install matplotlib
opencv-python          # pip install opencv-python
```

---

### Directory Setup

```
StyleConsensusLearning/
├── data/               # dataset images and label files
│   └── <DatasetName>/
│       ├── Images/
│       └── Labels/     # train.txt  test.txt  label.txt
├── features/           # extracted .npy feature files and .pkl style-set files
├── pretrainModels/     # pretrained VGG16 / ResNet50 weights
├── temp/               # checkpoint models saved during training
└── imgs/               # (optional) intermediate reconstructed images
```

> All `data/`, `features/`, `pretrainModels/`, and `backup/` files are available on Baidu YunPan.

---

### Usage

**1. Extract features** (run once per dataset/backbone combination)

Edit the dataset and backbone settings in `run_NEW.py`, then set `isImage=False` and choose `model_type`.

**2. Generate train / test label splits**

```bash
python utils/trainTestSplit.py
# or
python utils/trainTestValSplit.py
```

**3. Train**

```bash
python run_NEW.py          # uses feature mode (isImage=False) by default
```

Key arguments in `runStyleConsensusLearning()`:

| Parameter | Default | Description |
|---|---|---|
| `num_epochs` | 3000 | Training epochs |
| `batch_size` | 256 | Batch size |
| `isImage_` | False | `True` for raw images, `False` for pre-extracted features |
| `model_type` | `'resnet'` | `'vgg'` or `'resnet'` |

Checkpoints are saved to `temp/` whenever validation accuracy improves.

**4. Evaluate**

```bash
python run_test_NEW.py
```

**5. Predict a single image**

```bash
python style_predict.py
```

---

### Experiment Results vs. SOTA

|                      | Painting91 | Pandora | WikiArt3 | Arch      | FashionStyle14 | AVAStyle |
|----------------------|-----------|---------|----------|-----------|----------------|----------|
| VGG16                | 58.42     | 49.73   | 40.02    | 61.41     | 68.22          | 39.94    |
| VGG19                | 58.11     | 46.44   | 39.93    | 60.11     | 66.14          | 40.02    |
| ResNet50             | 64.93     | 51.65   | 47.01    | 65.12     | 71.13          | 40.05    |
| ResNet101            | 65.50     | 52.61   | 46.11    | 66.42     | 70.00          | 47.02    |
| InceptionV3          | 53.41     | 42.83   | 36.68    | 61.52     | 62.70          | 33.33    |
| DAE                  | 58.82     | 48.71   | 41.48    | 58.55     | 61.48          | 41.46    |
| SAE                  | 63.65     | 48.64   | 41.53    | 59.61     | 74.33          | 40.29    |
| SSCAE                | 64.07     | 49.38   | 43.65    | 60.48     | **75.02**      | 45.77    |
| DDS                  | 62.21     | 52.35   | 43.17    | /         | /              | /        |
| MCFFNet              | 66.60     | 51.39   | 45.51    | **66.12** | 68.38          | 42.69    |
| STACLE               | 60.41     | 55.80   | 47.21    | 60.81     | 64.47          | 46.38    |
| **TCFL+VGG16 TOP1**  | **67.39** | **56.67** | **47.85** | 65.57   | 71.67          | **47.22** |
| TCFL+VGG16 TOP2      | 83.19     | 74.92   | 66.99    | 78.12     | 85.69          | 63.92    |
| TCFL+VGG16 TOP3      | 92.27     | 84.52   | 78.22    | 85.05     | 91.11          | 75.74    |
| **TCFL+ResNet50 TOP1** | **69.12** | **56.98** | **51.62** | **69.03** | **77.17**   | **53.94** |
| TCFL+ResNet50 TOP2   | 85.29     | 76.15   | 69.11    | 81.82     | 87.44          | 68.68    |
| TCFL+ResNet50 TOP3   | 92.27     | 84.52   | 78.22    | 85.05     | 91.10          | 75.74    |

---

### Citation

If you find this work useful, please cite our paper:

```bibtex
@article{cui2025community,
  title     = {Community transferrable representation learning for image style classification},
  author    = {Cui, Jia, Shen Jinchen, Wei Jialin, Liu Shiyu, Ye Zhaojia, Luo Shijian, Qin Zhen},
  journal   = {ACM Transactions on Multimedia Computing, Communications and Applications},
  volume    = {21},
  number    = {6},
  pages     = {1--20},
  year      = {2025},
  publisher = {ACM}
}
```

> Cui, Jia, et al. "Community transferrable representation learning for image style classification." *ACM Transactions on Multimedia Computing, Communications and Applications* 21.6 (2025): 1–20.
