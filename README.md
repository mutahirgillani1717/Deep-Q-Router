# ⚡ Deep-Q Router: Interactive RL Agent

**Deep-Q Router** is a custom Reinforcement Learning environment and interactive dashboard built to demonstrate dynamic pathfinding and neural network optimization. It utilizes a Deep Q-Network (DQN) to discover mathematically optimal routing paths while dynamically avoiding user-defined network congestion.

---

## 🧠 Core Architecture

* **The Environment:** A custom grid-based Markov Decision Process (MDP) simulating network packet routing with distinct positive/negative reward structures.
* **The Agent:** A PyTorch-based Feed-Forward Neural Network featuring Experience Replay and Epsilon-Greedy exploration for stable learning.
* **Compute Optimization:** Implemented **End-of-Episode Replay Batching**, drastically reducing CPU backpropagation passes. This allows the model to train efficiently on standard hardware (e.g., AMD Ryzen 5) without relying on dedicated GPUs.

---

## 🎛️ Interactive Dashboard Features

Unlike static scripts, this project features a fully interactive Streamlit GUI:
* **Environment Builder:** Dynamically alter the network topology by moving the Start/End nodes and placing custom network congestion (obstacles). The system automatically prompts for retraining when the environment physics change.
* **AI Intelligence Query:** Select any node on the grid to "interrogate" the neural network. The dashboard extracts the raw PyTorch tensor weights to display the exact Q-Values (confidence scores) for moving Up, Down, Left, or Right.
* **Manual Override & Validation:** Allows users to manually drive the packet using a strict Manhattan Distance constraint, preventing illegal moves across the topology.

---

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mutahirgillani1717/Deep-Q-Router.git](https://github.com/mutahirgillani1717/Deep-Q-Router.git)
   cd Deep-Q-Router
2. Install dependencies:
pip install torch numpy streamlit

3. Launch the Dashboard:
streamlit run app.py

Author: Syed Mutahir Hussain

Academic Context: Final Year Computer Science | UET Taxila

Domain: Reinforcement Learning, Systems Optimization, & AI UI/UX