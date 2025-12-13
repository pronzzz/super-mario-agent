# Super Mario RL Agent 🎮

A state-of-the-art Deep Reinforcement Learning agent that learns to play Super Mario Bros using **Rainbow DQN** with advanced techniques including Spatial Transformer Networks and multi-branch architecture.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

This implementation includes state-of-the-art RL techniques that address common DQN problems:

### Complete Rainbow DQN Implementation

All **6 Rainbow DQN components** implemented:

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **Double DQN** | Separate action selection/evaluation | Reduces overestimation bias |
| **Dueling Architecture** | Separate value & advantage streams | Better state value estimation |
| **Distributional RL (C51)** | Model full value distribution | More stable learning |
| **Noisy Networks** | Learnable exploration noise | Better exploration than ε-greedy |
| **Prioritized Replay** | Sample important transitions | Improved sample efficiency |
| **Multi-step Returns** | n-step bootstrapping (n=3) | Faster credit assignment |

### Advanced Architecture Features

- **🎯 Spatial Transformer Network (STN)**: Learns to focus on relevant screen regions (enemies, gaps, power-ups)
- **🔗 Multi-Branch Architecture**: Combines visual features (CNN) with action history (MLP)
- **🧠 Attention Mechanism**: Adaptive spatial transformations for better feature extraction

## 🏗️ Architecture

```
Input: 4 stacked frames (84×84) + 8 action history
    ↓
┌─────────────────────────────────────────┐
│ VISUAL BRANCH                           │
│  STN → CNN (Nature DQN) → Features     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ ACTION HISTORY BRANCH                   │
│  Embedding → MLP → Features            │
└─────────────────────────────────────────┘
    ↓
    Fusion (Concatenate)
    ↓
┌─────────────────────────────────────────┐
│ DUELING HEADS (Noisy Layers)           │
│  Value Stream  → V(s)                  │
│  Advantage Stream → A(s,a)             │
└─────────────────────────────────────────┘
    ↓
Output: Q-value distribution (7 actions × 51 atoms)
```

### Network Details

**Visual Branch:**
- **Spatial Transformer Network**: 2D affine transformations
- **CNN Backbone**: Nature DQN architecture
  - Conv2d(4→32, kernel=8, stride=4)
  - Conv2d(32→64, kernel=4, stride=2)
  - Conv2d(64→64, kernel=3, stride=1)

**Action History Branch:**
- Embedding layer (7 actions → 32 dims)
- MLP (256→128→128)

**Dueling Heads with Noisy Layers:**
- Value Stream: NoisyLinear(512) → NoisyLinear(51)
- Advantage Stream: NoisyLinear(512) → NoisyLinear(7×51)
- Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/pronzzz/super-mario-agent.git
cd super-mario-agent

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training

```bash
python -m src.train
```

This will:
- Train the agent on Super Mario Bros World 1-1
- Save checkpoints every 500 episodes to `./mario_runs/<timestamp>/checkpoints/`
- Log metrics to TensorBoard

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=mario_runs
```

Open http://localhost:6006 to view:
- Episode rewards (cumulative and moving average)
- Loss curves
- Q-value estimates
- Episode lengths

### Play with Trained Agent

```bash
python play.py --checkpoint mario_runs/<timestamp>/checkpoints/mario_net_final.pth
```

Watch the agent play in real-time with visualization!

### Verify Installation

```bash
python verify.py
```

Runs unit tests for all components.

## ⚙️ Configuration

Edit `src/train.py` to modify training parameters:

```python
config = {
    'num_episodes': 10000,     # Total episodes
    'save_interval': 500,       # Checkpoint frequency
    'log_interval': 10,         # Console log frequency
    'device': 'cuda',           # 'cuda' or 'cpu'
    'save_dir': './mario_runs'  # Save directory
}
```

Key hyperparameters (in `src/agent.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2.5e-4 | Adam optimizer learning rate |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Batch Size | 32 | Samples per training step |
| Replay Buffer | 100,000 | Maximum stored transitions |
| Target Update | 10,000 steps | Sync frequency for target network |
| Learning Starts | 50,000 steps | Initial exploration period |
| Multi-step (n) | 3 | N-step return horizon |
| C51 Atoms | 51 | Distributional RL bins |
| Value Range | [-10, 10] | Distribution support |
| PER Alpha | 0.6 | Prioritization exponent |
| PER Beta | 0.4→1.0 | Importance sampling annealing |

## 📊 Expected Results

Training progression on World 1-1:

| Episodes | Behavior | Avg Reward |
|----------|----------|------------|
| 0-100 | Random exploration | ~100-200 |
| 100-500 | Learning to run right | ~300-600 |
| 500-1000 | Jumping over gaps | ~800-1500 |
| 1000-2000 | Consistent progress | ~1500-2500 |
| 2000-5000 | Level completion | ~2500-3000 |

**Convergence**: Reliable World 1-1 completion around **3000-5000 episodes** (~12-20 hours on CPU, ~3-6 hours on GPU).

## 🗂️ Project Structure

```
super-mario-agent/
├── src/
│   ├── __init__.py           # Package init
│   ├── wrappers.py           # Environment preprocessing
│   ├── model.py              # Rainbow DQN architecture
│   ├── replay.py             # Prioritized replay buffer
│   ├── agent.py              # Agent logic and learning
│   └── train.py              # Training loop
├── requirements.txt          # Dependencies
├── verify.py                 # Verification script
├── play.py                   # Play with trained agent
├── visualize.py              # Training visualization GUI
└── README.md                 # This file
```

## 🔍 Implementation Details

### Environment Preprocessing

1. **SkipFrame**: Repeat action for 4 frames (reduces computation by 4×)
2. **GrayScaleResize**: RGB → Grayscale + resize to 84×84
3. **FrameStack**: Stack last 4 frames (captures motion)
4. **ActionHistoryWrapper**: Track last 8 actions (custom wrapper)

Action Space: 7 simple movements
- NOOP, Right, Right+A, Right+B, Right+A+B, A, Left

### Distributional RL (C51)

Uses categorical cross-entropy instead of MSE:

```python
# Project target distribution onto 51-atom support
for each atom j:
    Tz = r + γⁿ * atom[j]
    Tz = clamp(Tz, v_min, v_max)
    
    # Linear interpolation to neighboring atoms
    b = (Tz - v_min) / Δz
    l, u = floor(b), ceil(b)
    
    # Distribute probability mass
    target_dist[l] += next_dist[j] * (u - b)
    target_dist[u] += next_dist[j] * (b - l)

loss = -Σ target_dist * log(current_dist)
```

### Prioritized Experience Replay

- **SumTree** data structure for O(log N) sampling
- Priorities = |TD error| + ε
- Importance sampling weights: w = (N * P(i))^(-β)
- Beta annealing from 0.4 to 1.0 over 100k frames

### Noisy Networks

Factorized Gaussian noise (Fortunato et al.):

```python
weight = weight_μ + weight_σ ⊙ ε
ε = sign(x) * √|x|, where x ~ N(0,1)
```

- Training: noisy weights (exploration)
- Evaluation: mean weights (deterministic)

## 📈 Visualization

The project includes two visualization tools:

### 1. TensorBoard (Real-time Training Metrics)

```bash
tensorboard --logdir=mario_runs
```

### 2. GUI Visualizer (Agent Performance)

```bash
python visualize.py --checkpoint <path_to_checkpoint>
```

Features:
- Live gameplay rendering
- Real-time Q-value distribution visualization
- Action history timeline
- Performance metrics dashboard

## 🎯 What Makes This Special

### Addresses Common DQN Problems

1. **Sample Inefficiency** → Rainbow components + n-step + PER
2. **Overestimation Bias** → Double DQN + distributional RL
3. **Poor Exploration** → Noisy networks (no ε-greedy)
4. **Training Instability** → Target network + gradient clipping + dueling
5. **Limited Attention** → Spatial Transformer Network

### Production-Ready Code

- ✅ Comprehensive documentation
- ✅ Type hints and docstrings
- ✅ Modular design
- ✅ TensorBoard integration
- ✅ Checkpoint management
- ✅ Verification tests

## 🔮 Future Enhancements

Potential extensions:

1. **Curiosity-Driven Exploration**: Intrinsic rewards (RND, ICM)
2. **Data Augmentation**: Random crop, color jitter (RAD)
3. **Distributed Training**: IMPALA/Ape-X for parallel actors
4. **Generalization**: Multi-level training and transfer learning
5. **Recurrent Networks**: LSTM/GRU for partial observability
6. **Imitation Learning**: Pretrain on human demonstrations

## 📚 References

This implementation is based on:

1. **Rainbow DQN**: [Hessel et al., 2018](https://arxiv.org/abs/1710.02298)
2. **Spatial Transformer Networks**: [Jaderberg et al., 2015](https://arxiv.org/abs/1506.02025)
3. **Human-level control through deep RL**: [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
4. **Prioritized Experience Replay**: [Schaul et al., 2016](https://arxiv.org/abs/1511.05952)
5. **Noisy Networks**: [Fortunato et al., 2018](https://arxiv.org/abs/1706.10295)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💡 Acknowledgments

- OpenAI Gym and gym-super-mario-bros for the environment
- PyTorch team for the deep learning framework
- DeepMind for Rainbow DQN research

---

**Built with ❤️ using PyTorch and reinforcement learning**
