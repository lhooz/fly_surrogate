# Bio-Fluid LBM Surrogate: Hybrid Physics-ML Aerodynamics

**Bio-Fluid LBM Surrogate** is a high-fidelity simulation framework that bridges the gap between accurate fluid dynamics and fast neural inference. It utilizes **Taichi** to run a Lattice Boltzmann Method (LBM) fluid simulation and **JAX** to train a real-time neural surrogate model on the generated data.



Key features include:
* **Hybrid Architecture:** Seamless integration of GPU-accelerated LBM (via Taichi) with differentiable learning (via JAX).
* **Immersed Boundary Method:** Robust fluid-structure interaction for handling complex flapping wing kinematics.
* **Neural Surrogate:** A ResNet-based architecture that learns to predict unsteady aerodynamic forces (Lift/Drag) from kinematic history.
* **Online Training:** Simultaneous data generation and model training loop for continuous learning.
* **Global-Local Decomposition:** Decouples global body jitter from local wing kinematics for generalized learning.

### 🎓 Try it now
Run the full training demo in your browser with zero setup (requires T4 GPU runtime):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhooz/fly_surrogate/blob/main/notebooks/train_surrogate.ipynb)

## 📂 Project Structure

```text
fly_surrogate/                  <-- Repository Root
├── environment_engine_taichi.py # High-performance LBM Fluid Solver (Taichi)
├── train_surrogate_jax.py       # Main Training Loop & Kinematics Generator
├── pyproject.toml               # Dependency & Project Configuration
├── README.md                    # Project Documentation
└── checkpoints/                 # Saved models and visualizations
    ├── checkpoint.pkl
    └── viz_cycle_0000.gif

```

---

## 🚀 Installation

### 1. Local Installation

Clone the repository and install dependencies using the configuration in `pyproject.toml`:

```bash
git clone [https://github.com/YOUR_USERNAME/fly_surrogate.git](https://github.com/YOUR_USERNAME/fly_surrogate.git)
cd fly_surrogate
pip install -e .

```

### 2. Google Colab Installation

In a Colab cell, you can install directly from your repository or local folder:

```python
# Mount Drive if your code is stored there
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/fly_surrogate

# Install dependencies (includes Taichi & JAX)
!pip install -e .

```

---

## 🏋️‍♂️ Training

The training script runs the Taichi fluid simulation to generate ground-truth aerodynamic data, which is immediately fed into a JAX neural network for training.

![CNN Architecture](images/cnn_architecture.png)

**Run on GPU (Recommended):**
This script is optimized for Google Colab (T4 GPU). It limits JAX and Taichi memory usage to prevent conflicts.

```bash
python train_surrogate_jax.py

```

**Training Output:**

* **Logs:** Real-time updates on Cycle count, Loss, and Flight Kinematics.
* **Visualizations:** Every 100 cycles, a `.gif` is generated in `checkpoints/` showing the wing interacting with the fluid vorticity field.
* **Checkpoints:** The model weights are saved automatically as `checkpoint.pkl`.

---

## ⚙️ Configuration

Key simulation parameters are defined in `environment_engine_taichi.py`:

* **Grid Resolution:** 500 x 500 (LBM Lattice)
* **Reynolds Number:** Controlled via `NU_SI` (Viscosity) and `PHYS_SIZE`.
* **Time Step:** 3.0e-6 s (Fluid Solver) vs 3.0e-5 s (Surrogate Model).

## 📦 Dependencies

* [Taichi](https://github.com/taichi-dev/taichi) - Parallel Fluid Solver
* [JAX](https://github.com/google/jax) - Differentiable Programming
* [DM-Haiku](https://github.com/deepmind/dm-haiku) - Neural Network Library
* [Optax](https://github.com/deepmind/optax) - Optimization
* [Matplotlib](https://matplotlib.org/) - Visualization

## 📄 License

This project is open-source. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.