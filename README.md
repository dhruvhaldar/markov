# Markov

Markov is a web-based Deep Reinforcement Learning platform built for EL2805 Reinforcement Learning. It provides a unified, visual interface to train agents using both classical tabular methods (Dynamic Programming, SARSA) and state-of-the-art policy gradient algorithms (PPO, DDPG).

The tool features a Dark Space Material Design (Flat 2.0) UI, using strict elevation shadows and flat geometry to create a clean, modern deep-space analytics dashboard.

## 📚 Syllabus & Lab Mapping (EL2805)

This project implements the specific environments and algorithms requested in the course labs:

| Module | Course Topic | Implemented Features |
|---|---|---|
| Tabular RL | Lab 1: Maze & Minotaur | Value Iteration, Q-Learning, and SARSA ($\epsilon$-greedy) for discrete gridworlds. |
| Linear Approx | Lab 1: Mountain Car | Sarsa($\lambda$) with a p=2 Fourier Basis and Nesterov accelerated SGD. |
| DQN | Lab 2: Lunar Lander (Discrete) | Deep Q-Network with an Experience Replay Buffer and delayed Target Network updates. |
| DDPG | Lab 2: Lunar Lander (Cont.) | Actor-Critic architecture with Ornstein-Uhlenbeck noise for continuous action exploration. |
| PPO | Lab 2: Trust Region Methods | Proximal Policy Optimization using clipped surrogate objectives to guarantee monotonic improvement. |

## 🚀 Deployment (Vercel)

Markov is designed to run as a serverless training and inference engine.

1. Fork this repository.
2. Deploy to Vercel (Python runtime is auto-detected).
3. Access the RL Dashboard at `https://your-markov.vercel.app`.

## 📊 Visualizations & Artifacts

### 1. Lunar Lander: Deep Q-Network (DQN) vs PPO
Visualizing the learning curve: Total Episodic Reward over Time.

**Code Snippet (PyTorch):**
```python
# DQN Target Network Update Logic [cite: 164]
if steps % C == 0:
    target_network.load_state_dict(main_network.state_dict())
```

**Figure 1:** DQN Learning Curve. The agent learns to land between the two flags (coordinates 0,0). The graph tracks the reward, showing convergence toward the 200-point threshold required to solve the environment. Displayed using Material Design elevated cards.

### 2. Mountain Car: Fourier Basis & Sarsa($\lambda$)
A visualization of the state-value function $V(s)$ over the continuous state space (Position vs. Velocity).

**Code Snippet (Sarsa Eligibility Trace):**
```python
# Eligibility Trace Update [cite: 613, 618]
z_a = gamma * lambda_ * z_a + grad_Q
w = w + alpha * delta * z_a
```

**Figure 2:** Mountain Car Value Function. By utilizing a Fourier Basis, the agent approximates the continuous state space without discretizing it. The eligibility trace ($z$) allows the agent to correctly assign credit to actions taken long before the sparse +1 reward is achieved at the hilltop.

### 3. The Minotaur Maze: Dynamic Programming
A heatmap of the optimal policy derived via Value Iteration.

**Figure 3:** Minotaur Evasion Policy. The gridworld visualizer plots the optimal path to exit the maze while minimizing the probability of intersecting the random-walking Minotaur. The heatmap shows the value of each state $V(s)$ updating iteratively until convergence.
