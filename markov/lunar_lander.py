import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=100000, batch_size=64, target_update_freq=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.main_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.main_network(state_tensor)
        return q_vals.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_vals = self.main_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_vals = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_vals = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = F.mse_loss(q_vals, target_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        # DQN Target Network Update Logic [cite: 164]
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        return loss.item()

def train_dqn(episodes=100, max_steps=1000, batch_size=64):
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, batch_size=batch_size)

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

    env.close()
    return agent, rewards_history

# --- DDPG Implementation ---

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.fc1(state))
        q = F.relu(self.fc2(torch.cat([q, action], 1)))
        return self.fc3(q)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim)

        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).numpy()[0]
        self.actor.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Target Updates
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

# --- PPO Implementation ---

from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        # Actor
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, action_dim)

        # Critic
        self.critic_fc1 = nn.Linear(state_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_probs = F.softmax(self.actor_fc3(x), dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        # Actor
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_probs = F.softmax(self.actor_fc3(x), dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Critic
        v = F.relu(self.critic_fc1(state))
        v = F.relu(self.critic_fc2(v))
        state_values = self.critic_fc3(v)

        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)
        return action, action_logprob.item()

    def update(self, memory):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)
            advantages = rewards - state_values.detach()

            ratios = torch.exp(logprobs - old_logprobs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class PPOMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train_ppo(episodes=100, max_steps=1000, update_timestep=2000):
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    memory = PPOMemory()

    time_step = 0
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            time_step += 1

            action, logprob = agent.choose_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            memory.states.append(torch.FloatTensor(state).unsqueeze(0))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(torch.tensor(logprob))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            total_reward += reward

            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0

            if done:
                break

        rewards_history.append(total_reward)

    env.close()
    return agent, rewards_history