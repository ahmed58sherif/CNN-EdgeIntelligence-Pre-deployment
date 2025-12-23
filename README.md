# 🧠 CNN Edge Intelligence: Software-Hardware Co-Design for LeNet-5 Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14](https://img.shields.io/badge/tensorflow-2.14-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: ICEENG 2026](https://img.shields.io/badge/Paper-ICEENG%202026-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📋 Overview

This repository contains **comprehensive Python-based empirical research** evaluating CNN optimization techniques for edge inference on resource-constrained devices. We systematically analyze **quantization, pruning, and mixed-precision** methods applied to LeNet-5 for MNIST classification, complementing hardware-level FPGA acceleration work with software-level pre-deployment analysis.

### ⚡ Key Results

| Technique | Accuracy | Compression | DSP Util. | Status |
|-----------|----------|-------------|-----------|--------|
| **Baseline (Float32)** | 95.77% | — | 86 | Baseline |
| **Q4.4 (8-bit)** | 93.88% | -50.6% | 52 | Aggressive |
| **50% Pruning** | 95.74% | -0.03% | 68 | Efficient |
| **Mixed-Precision** | 95.95% | -13.8% | 56 | ⭐ **Optimal** |
| **Pruning + Q4.4** | 93.62% | -75.3% | 45 | Extreme |

---

## 🎯 Features

✅ **Comprehensive Evaluation Framework**
- Per-class precision, recall, F1-score metrics
- Confusion matrix analysis for per-digit robustness
- Pareto frontier visualization

✅ **Production-Ready Implementation**
- Full reproducibility: Python 3.10, TensorFlow 2.14, NumPy 1.24
- CPU-only execution (no GPU dependency)
- Hyperparameters fully documented

✅ **Real-Time Feasibility Analysis**
- Rate Monotonic Scheduling (RMS) verification
- WCET (worst-case execution time) profiling
- Xilinx Zynq-7000 resource estimation (5.6% MAPE)

✅ **Professional Visualization**
- Training convergence curves
- Accuracy-compression Pareto frontiers
- RMS Gantt scheduling diagrams
- Per-class confusion matrices

---

## 📁 Repository Structure

```
CNN-EdgeIntelligence/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── 🔬 notebooks/
│   └── LeNet_V3.ipynb  # Interactive Jupyter notebook
│
│
├── 📈 results & figures/
│   ├── fig_training_curves.png              # Training dynamics
│   ├── fig_cm_lenet_mnist.png               # Confusion matrix
│   ├── fig_acc_pareto.png                   # Pareto frontier ⭐
│   ├── fig_ablation_all_methods.png         # Technique comparison
│   ├── fig_accuracy_vs_storage.png          # Compression tradeoff
│   ├── fig_rtos_rms_gantt.png               # RMS scheduling
│   └── fig_enhancements_vs_q16.png          # Technique impact
│
├── 🎓 paper/
│   ├── CNN_EdgeIntelligence_ICEENG_2026.pdf  # Published paper

```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Jupyter Notebook** (for interactive analysis)
- **RAM:** 16GB recommended
- **Storage:** 2GB for datasets + models

### Installation

```bash
# Clone repository
git clone https://github.com/ahmed58sherif/CNN-EdgeIntelligence-Pre-deployment.git
cd CNN-EdgeIntelligence

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print(tensorflow.__version__)"  # Should print 2.14.0
```

### Run the Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/CNN_EdgeIntelligence_Analysis.ipynb

# Select "Kernel → Restart & Run All" to reproduce all results
# Expected runtime: ~15 minutes (CPU-only)
```

### Expected Output

The notebook generates:
- ✅ Baseline model (95.77% accuracy)
- ✅ Q4.4 quantization (93.88%)
- ✅ 50% magnitude pruning (95.74%)
- ✅ Mixed-precision (95.95% — best accuracy!)
- ✅ Combined optimization (75.3% compression)
- ✅ 6 publication-quality figures
- ✅ Per-class sensitivity analysis

---

## 📊 Key Findings

### 1️⃣ **Mixed-Precision Dominates**
Assigning Q4.4 (8-bit) to convolutional layers and Q6.6 (12-bit) to fully-connected layers:
- **Improves accuracy by 0.18%** (95.77% → 95.95%)
- **Reduces DSP utilization by 35%** (86 → 56)
- **Optimal accuracy-efficiency tradeoff**

### 2️⃣ **Weight Distribution Explains Quantization Loss**
LeNet-5 weights concentrate in [-0.3, 0.3], utilizing only 7.5% of Q4.4's [-8, 8] range:
- Quantization step mismatch causes 1.89% accuracy loss (vs. typical 0.5-1.5%)
- Post-training rescaling (s ≈ 26.67) recovers 0.8% accuracy
- QAT can recover 0.82% additional improvement (→ 94.7%)

### 3️⃣ **Pruning as Implicit Regularization**
50% magnitude-based pruning removes 2.4M MACs from Conv2 layer with **minimal loss**:
- Accuracy: 95.77% → 95.74% (**-0.03% only!**)
- Demonstrates parameter redundancy
- Consistent with "lottery ticket hypothesis"

### 4️⃣ **Real-Time Feasibility Guaranteed**
RMS scheduling verification ensures hard real-time deadlines:
- CPU utilization: **0.0272 ≪ 0.757 bound**
- Inference WCET: **1.07 ms** (65% reduction vs. float32)
- Spare capacity: **97.28%** for aperiodic events (obstacle detection, emergencies)

---

## 📈 Evaluation Metrics

The evaluation framework provides:

### Per-Class Metrics
```python
Precision = TP / (TP + FP)     # False positive rate
Recall    = TP / (TP + FN)     # False negative rate  
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

### Hardware Metrics
- **DSP Utilization**: Digital signal processor usage (%)
- **BRAM**: Block RAM footprint (KB)
- **Latency**: Inference time per image (ms)
- **Throughput**: Frames per second (FPS)

### Compression Metrics
- **Storage Reduction**: Model size decrease (%)
- **Parameter Reduction**: Weight count decrease (%)
- **MAC Reduction**: Computational complexity decrease (%)

---

## 🔬 Methodology

### Baseline Model
- **Architecture**: LeNet-5 (61.7K parameters)
- **Training**: SGD, LR=0.01, batch=64, 5 epochs
- **Dataset**: MNIST (60K training, 10K test)
- **Test Accuracy**: 95.77%

### Quantization Schemes
1. **Q8.8 (16-bit)**: Emulates prior FPGA work, matches hardware baseline
2. **Q4.4 (8-bit)**: Aggressive compression, -1.89% accuracy loss
3. **Q6.6 (12-bit)**: For fully-connected layers in mixed-precision

### Pruning Strategy
- **Method**: Magnitude-based structured pruning
- **Target**: Conv2 layer (93.3% of MACs)
- **Ratio**: 50% (reduces 2.4M → 1.2M MACs)
- **Effect**: -0.03% accuracy loss (near-lossless!)

### Mixed-Precision Assignment
```python
Conv layers → Q4.4 (8-bit)   # Spatial averaging tolerance
FC layers   → Q6.6 (12-bit)  # Individual feature precision
Result: +0.18% accuracy with 35% DSP reduction
```

---

## 📚 Citation

If you use this code or results in your research, please cite:

```bibtex
@inproceedings{Abdelazem2026CNN,
  author = {Abdelazem, Ahmed Sherif and Elsedfy, Mohamed Omar Mahmoud and Elshafey, Mohamed Abdelmoneim Taha},
  title = {Software-Hardware Co-Design for Low-Power Edge Inference: Pre-Deployment Compression of CNN Models},
  booktitle = {Proceedings of the 2026 IEEE Conference on Innovations in Intelligent Computing and Cybersecurity (ICEENG)},
  year = {2026},
  organization = {IEEE},
  note = {Under review}
}
```

---

## 🔗 Related Work

This work complements:
- **[Liang et al., 2024]**: Hardware FPGA acceleration (LeNet-5 PIPELINE accelerator, 70× speedup)
- **[Dong et al., 2023]**: Mixed-precision quantization theory
- **[Cheng et al., 2023]**: Neural network pruning survey
- **[Butt et al., 2023]**: RTOS rate monotonic scheduling

---

## 🚦 Deployment Recommendations

Choose configuration based on your constraints:

| Scenario | Method | Accuracy | Storage | DSP | Use Case |
|----------|--------|----------|---------|-----|----------|
| **Accuracy-Critical** | Mixed-Precision | 95.95% | 213 KB | 56 | Autonomous vehicles, medical imaging |
| **Balanced** | 50% Pruning | 95.74% | 247 KB | 68 | Smart cities, industrial IoT |
| **Resource-Constrained** | Q4.4 Only | 93.88% | 122 KB | 52 | Wearables, mobile devices |
| **Extreme Compression** | Prune + Q4.4 | 93.62% | 61 KB | 45 | 64KB SRAM microcontrollers |

---

## 🔮 Future Work

1. **HLS Implementation**: PIPELINE pragmas on Xilinx Zynq
2. **Larger Networks**: MobileNetV2, ResNet-18, EfficientNet
3. **Quantization-Aware Training**: Recover 1-2% QAT loss
4. **Dynamic Precision**: Runtime bit-width adjustment
5. **Knowledge Distillation**: Extreme compression for embedded systems
6. **Sparse Convolution**: GPU-accelerated sparse implementations


---

## 📖 How to Use This Repository

### For Researchers
```bash
# Reproduce paper results
jupyter notebook notebooks/LeNet_V3.ipynb

# Modify hyperparameters
# Edit: pruning_ratio = 0.5  (line 45)
#       quantization_bits = 4  (line 67)

# Generate custom metrics
python scripts/evaluate_custom.py --method mixed_precision --output results/
```

### For Practitioners
```bash
# Load pre-trained mixed-precision model
import tensorflow as tf
model = tf.keras.models.load_model('models/lenet5_mixed_precision.h5')

# Inference on new data
predictions = model.predict(new_images)
```

### For Educators
Use figures in presentations:
- `fig_training_curves.png` — Show CNN convergence
- `fig_acc_pareto.png` — Illustrate accuracy-efficiency tradeoffs
- `fig_rtos_rms_gantt.png` — Real-time OS scheduling concepts

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model
ARCHITECTURE = 'lenet5'
INPUT_SHAPE = (28, 28, 1)

# Training
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# Quantization
QUANT_BITS = {
    'conv': 4,      # Q4.4 for Conv layers
    'fc': 6         # Q6.6 for FC layers
}

# Pruning
PRUNING_RATIO = 0.50  # 50% magnitude-based
TARGET_LAYER = 'Conv2'

# Hardware
ZYNQ_FREQUENCY = 650  # MHz
DSP_COUNT = 360       # Zynq-7000 DSPs
```

---

## 📞 Support & Issues

- **Email**: ahmed58sherif@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

**Citation**: If you modify or extend this work, please maintain the MIT license and acknowledge the original authors.

---

## ✨ Acknowledgments

- **Dataset**: MNIST (LeCun et al., 1999)
- **Baseline Hardware**: Xilinx Zynq-7000 XC7Z020 (Liang et al., 2024)
- **Reference Implementations**: TensorFlow/Keras community

---

## 🔐 Reproducibility Statement

✅ **Fully Reproducible**:
- CPU-only execution (no GPU variance)
- Fixed random seeds (42)
- All hyperparameters documented
- ~15 minutes to reproduce all results

---

## 📊 Citation Statistics

```
GitHub Stars: ⭐⭐⭐⭐⭐
GitHub Forks: 📌
Paper Citations: 📈
```

---

**Last Updated**: May 2026
**Status**: Active development for ICEENG 2026 publication
