# Bio-Fluid LBM Surrogate: Hybrid Physics-ML Aerodynamics

![Inference Demo](images/inference.gif)
*Side-by-side validation on unseen kinematic data. **Left:** Real-time fluid vorticity field (Background) with instantaneous force vectors (Green=Ground Truth, Pink=Prediction). **Right:** Time-history of Lift and Drag forces showing the surrogate model tracking the CFD solver with high fidelity.*

**Bio-Fluid LBM Surrogate** is a high-fidelity simulation framework that bridges the gap between accurate fluid dynamics and fast neural inference. It utilizes **Taichi** to run a Lattice Boltzmann Method (LBM) fluid simulation and **JAX** to train a real-time neural surrogate model on the generated data.

### 🌟 Key Features

* **Hybrid Compute Architecture:** Seamless integration of **GPU-accelerated LBM** (via Taichi) for physics and **JAX** for differentiable learning, managing memory contention on a single device.
* **Bio-Relevant Aerodynamics:** Implements a **D2Q9 Lattice Boltzmann** solver with LES stability, capturing complex **unsteady vortex shedding** at low-to-moderate Reynolds numbers ($Re \sim 100-1000$).
* **Robust Fluid-Structure Interaction (FSI):** Features a robust **Brinkman Penalization** immersed boundary method to map Lagrangian structure points to the Eulerian fluid grid without complex remeshing.
* **Infinite Data Pipeline (Online Learning):** Generates physics data on-the-fly during training, eliminating static dataset storage and allowing the model to learn from a continuously varying state space.
* **Geometric Invariance:** Decouples global body jitter from local wing kinematics (Global-Local Decomposition), allowing the neural network to learn efficient, generalized aerodynamic laws independent of position.
* **Compliant "Ghost" Coupling:** Models structural flexibility and actuator lag via a spring-damper coupling between the control target and the physical wing, preventing numerical instability during aggressive maneuvers.

### 🎓 Try it now

Run the full training demo in your browser with zero setup (requires T4 GPU runtime):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhooz/fly_surrogate/blob/main/notebooks/train_surrogate.ipynb)

---

## 🧠 Neural Architecture

The surrogate model uses a **1D ResNet** designed to capture spatial correlations along the wing chord. Unlike 2D image models, this architecture respects the topological sequence of the discretization points and processes kinematic history directly.

```mermaid
graph LR
    subgraph "Inputs (N_pts x 6)"
    A[Position<br/>(x, y)]
    B[Velocity<br/>(u, v)]
    C[Acceleration<br/>(ax, ay)]
    end

    subgraph "FluidSurrogateResNet (JAX)"
    D[Concatenate]
    E[Conv1D<br/>Feature Projection]
    F[ResNet Block 1<br/>(GELU + Skip)]
    G[ResNet Block 2<br/>(GELU + Skip)]
    H[ResNet Block 3<br/>(GELU + Skip)]
    I[Conv1D<br/>Readout Head]
    end

    subgraph "Outputs (N_pts x 2)"
    J[Aerodynamic Forces<br/>(Lift, Drag)]
    end

    A & B & C --> D
    D --> E --> F --> G --> H --> I
    I --> J

    style A fill:#e1f5fe,stroke:#01579b
    style B fill:#e1f5fe,stroke:#01579b
    style C fill:#e1f5fe,stroke:#01579b
    style J fill:#fce4ec,stroke:#880e4f

```

---

## 📂 Project Structure

```text
fly_surrogate/                 <-- Repository Root
├── environment_engine.py      # Taichi LBM Solver (LES + FSI)
├── train_surrogate.py         # Online Teacher-Student Training Loop
├── inference_surrogate.py     # Validation Script (Generates .gif)
├── pyproject.toml             # Dependencies
├── images/                    # Documentation Assets
│   └── inference.gif          # Validation Animation
└── checkpoints/               # Saved models
    └── checkpoint.pkl

```

---

## 🚀 Installation

### 1. Local Installation

Clone the repository and install dependencies using the configuration in `pyproject.toml`:

```bash
git clone [https://github.com/lhooz/fly_surrogate.git](https://github.com/lhooz/fly_surrogate.git)
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

## 🏋️‍♂️ Training & Inference

**1. Train the Model**
The training script runs the Taichi fluid simulation to generate ground-truth aerodynamic data, which is immediately fed into the JAX neural network.

```bash
python train_surrogate.py

```

**2. Validate Performance**
To generate the validation GIF shown above, run the inference script. It tests the model on random kinematic parameters outside the training set to ensure generalization.

```bash
python inference_surrogate.py

```

---

## ⚙️ Configuration

Key simulation parameters are defined in `environment_engine.py`:

* **Grid Resolution:** 500 x 500 (LBM Lattice)
* **Physics:** D2Q9 LBM with Smagorinsky Sub-grid Model ()
* **Time Step:**  s (Fluid Solver) vs  s (Surrogate Model)

## 📦 Dependencies

* [Taichi](https://github.com/taichi-dev/taichi) - Parallel Fluid Solver
* [JAX](https://github.com/google/jax) - Differentiable Programming
* [DM-Haiku](https://github.com/deepmind/dm-haiku) - Neural Network Library
* [Optax](https://github.com/deepmind/optax) - Optimization
* [Matplotlib](https://matplotlib.org/) - Visualization

## 📄 License

This project is open-source. See the LICENSE file for details.