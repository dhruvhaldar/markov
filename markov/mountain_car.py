import numpy as np
import gym
import itertools

class FourierBasis:
    def __init__(self, state_dim, order=2):
        self.order = order
        self.state_dim = state_dim
        # Generate all combinations of [0, 1, ..., order] for each state dimension
        c = [list(range(order + 1)) for _ in range(state_dim)]
        self.c = np.array(list(itertools.product(*c)))

    def get_features(self, state):
        # state must be normalized to [0, 1] before calling this
        return np.cos(np.pi * np.dot(self.c, state))

class SarsaLambdaNesterov:
    def __init__(self, state_dim, action_dim, order=2, alpha=0.01, gamma=0.99, lambda_=0.9, momentum=0.9):
        self.basis = FourierBasis(state_dim, order)
        self.num_features = len(self.basis.c)
        self.action_dim = action_dim

        # Weights for each action
        self.w = np.zeros((action_dim, self.num_features))

        # Nesterov momentum variables
        self.velocity = np.zeros((action_dim, self.num_features))

        # Eligibility traces
        self.z = np.zeros((action_dim, self.num_features))

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.momentum = momentum

    def get_q(self, state, action=None):
        features = self.basis.get_features(state)
        q_vals = np.dot(self.w, features)
        if action is not None:
            return q_vals[action]
        return q_vals

    def choose_action(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        q_vals = self.get_q(state)
        return np.argmax(q_vals)

    def reset_traces(self):
        self.z.fill(0)

    def update(self, state, action, reward, next_state, next_action, done):
        features = self.basis.get_features(state)
        next_features = self.basis.get_features(next_state)

        q_current = np.dot(self.w[action], features)

        if done:
            q_next = 0.0
        else:
            q_next = np.dot(self.w[next_action], next_features)

        td_error = reward + self.gamma * q_next - q_current

        # Eligibility Trace Update [cite: 613, 618]
        self.z *= self.gamma * self.lambda_
        self.z[action] += features

        # Nesterov Accelerated Gradient Descent
        grad = -td_error * self.z

        # Update velocity
        v_prev = self.velocity.copy()
        self.velocity = self.momentum * self.velocity - self.alpha * grad

        # Update weights with Nesterov Lookahead
        self.w += -self.momentum * v_prev + (1 + self.momentum) * self.velocity

def train_mountain_car(episodes=100, max_steps=1000, render=False):
    env = gym.make('MountainCar-v0')
    agent = SarsaLambdaNesterov(state_dim=2, action_dim=3, order=2, alpha=0.01)

    state_low = env.observation_space.low
    state_high = env.observation_space.high
    state_range = state_high - state_low

    def normalize(state):
        return (state - state_low) / state_range

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = normalize(state)

        action = agent.choose_action(state, epsilon=0.1)
        agent.reset_traces()

        total_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = normalize(next_state)

            next_action = agent.choose_action(next_state, epsilon=0.1)

            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

    env.close()
    return agent, rewards_history