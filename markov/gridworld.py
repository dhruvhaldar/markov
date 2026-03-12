import numpy as np
import random

class MazeMDP:
    def __init__(self, maze_grid, minotaur_start, player_start):
        """
        maze_grid: 2D array-like, 0 for empty, 1 for wall.
        """
        self.maze = np.array(maze_grid)
        self.rows, self.cols = self.maze.shape
        self.minotaur_start = tuple(minotaur_start)
        self.player_start = tuple(player_start)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)] # Right, Left, Down, Up, Stay
        self.action_space = list(range(len(self.actions)))
        self.reset()

    def reset(self):
        self.player_pos = self.player_start
        self.minotaur_pos = self.minotaur_start
        return self._get_state()

    def _get_state(self):
        return (self.player_pos, self.minotaur_pos)

    def step(self, action_idx):
        action = self.actions[action_idx]

        # Player moves
        new_player_pos = (self.player_pos[0] + action[0], self.player_pos[1] + action[1])
        if 0 <= new_player_pos[0] < self.rows and 0 <= new_player_pos[1] < self.cols and self.maze[new_player_pos] == 0:
            self.player_pos = new_player_pos

        # Minotaur moves (random walk)
        minotaur_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        valid_moves = []
        for m in minotaur_moves:
            new_pos = (self.minotaur_pos[0] + m[0], self.minotaur_pos[1] + m[1])
            if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                valid_moves.append(new_pos)
        if valid_moves:
            self.minotaur_pos = random.choice(valid_moves)

        # Check termination
        if self.player_pos == self.minotaur_pos:
            return self._get_state(), -100, True # Eaten

        # Exit condition (e.g. at bottom right)
        if self.player_pos == (self.rows - 1, self.cols - 1):
            return self._get_state(), 100, True # Escaped

        return self._get_state(), -1, False # Step penalty

class RobbingBanksMDP:
    def __init__(self, grid_size=(5, 5), police_start=(2, 2), player_start=(0, 0), bank_pos=(4, 4)):
        self.grid_size = grid_size
        self.police_start = tuple(police_start)
        self.player_start = tuple(player_start)
        self.bank_pos = tuple(bank_pos)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.action_space = list(range(len(self.actions)))
        self.reset()

    def reset(self):
        self.player_pos = self.player_start
        self.police_pos = self.police_start
        return self._get_state()

    def _get_state(self):
        return (self.player_pos, self.police_pos)

    def step(self, action_idx):
        action = self.actions[action_idx]

        # Player moves
        new_player_pos = (self.player_pos[0] + action[0], self.player_pos[1] + action[1])
        if 0 <= new_player_pos[0] < self.grid_size[0] and 0 <= new_player_pos[1] < self.grid_size[1]:
            self.player_pos = new_player_pos

        # Police moves towards player
        police_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_dist = float('inf')
        best_moves = [self.police_pos]
        for m in police_moves:
            new_pos = (self.police_pos[0] + m[0], self.police_pos[1] + m[1])
            if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
                dist = abs(new_pos[0] - self.player_pos[0]) + abs(new_pos[1] - self.player_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_moves = [new_pos]
                elif dist == best_dist:
                    best_moves.append(new_pos)
        self.police_pos = random.choice(best_moves)

        # Check termination
        if self.player_pos == self.police_pos:
            return self._get_state(), -50, True # Caught

        reward = 10 if self.player_pos == self.bank_pos else 0
        return self._get_state(), reward, False

def value_iteration(mdp, gamma=0.99, theta=1e-6):
    """
    Standard Value Iteration.
    Assuming mdp has states defined. Here we just build states for Maze or Bank.
    """
    if isinstance(mdp, MazeMDP):
        states = []
        for r in range(mdp.rows):
            for c in range(mdp.cols):
                if mdp.maze[r, c] == 0:
                    for mr in range(mdp.rows):
                        for mc in range(mdp.cols):
                            states.append(((r, c), (mr, mc)))
    elif isinstance(mdp, RobbingBanksMDP):
        states = []
        for r in range(mdp.grid_size[0]):
            for c in range(mdp.grid_size[1]):
                for pr in range(mdp.grid_size[0]):
                    for pc in range(mdp.grid_size[1]):
                        states.append(((r, c), (pr, pc)))

    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}

    while True:
        delta = 0
        for s in states:
            if isinstance(mdp, MazeMDP):
                if s[0] == s[1]:
                    V[s] = -100
                    continue
                if s[0] == (mdp.rows - 1, mdp.cols - 1):
                    V[s] = 100
                    continue
            else:
                if s[0] == s[1]:
                    V[s] = -50
                    continue

            v = V[s]
            max_val = float('-inf')
            best_a = 0

            for a in mdp.action_space:
                expected_val = 0

                # Player moves
                p_pos = s[0]
                action = mdp.actions[a]
                new_p = (p_pos[0] + action[0], p_pos[1] + action[1])

                if isinstance(mdp, MazeMDP):
                    if not (0 <= new_p[0] < mdp.rows and 0 <= new_p[1] < mdp.cols and mdp.maze[new_p] == 0):
                        new_p = p_pos # Hit wall

                    # Minotaur moves
                    m_pos = s[1]
                    m_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    valid_m = []
                    for m in m_moves:
                        new_m = (m_pos[0] + m[0], m_pos[1] + m[1])
                        if 0 <= new_m[0] < mdp.rows and 0 <= new_m[1] < mdp.cols:
                            valid_m.append(new_m)

                    if not valid_m:
                        valid_m = [m_pos]

                    prob = 1.0 / len(valid_m)
                    for new_m in valid_m:
                        if new_p == new_m:
                            reward = -100
                            next_v = 0 # terminal
                        elif new_p == (mdp.rows - 1, mdp.cols - 1):
                            reward = 100
                            next_v = 0 # terminal
                        else:
                            reward = -1
                            next_v = V.get((new_p, new_m), 0)

                        expected_val += prob * (reward + gamma * next_v)

                elif isinstance(mdp, RobbingBanksMDP):
                    if not (0 <= new_p[0] < mdp.grid_size[0] and 0 <= new_p[1] < mdp.grid_size[1]):
                        new_p = p_pos

                    # Police moves
                    pol_pos = s[1]
                    pol_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    best_dist = float('inf')
                    valid_pol = [pol_pos]
                    for m in pol_moves:
                        new_pol = (pol_pos[0] + m[0], pol_pos[1] + m[1])
                        if 0 <= new_pol[0] < mdp.grid_size[0] and 0 <= new_pol[1] < mdp.grid_size[1]:
                            dist = abs(new_pol[0] - new_p[0]) + abs(new_pol[1] - new_p[1])
                            if dist < best_dist:
                                best_dist = dist
                                valid_pol = [new_pol]
                            elif dist == best_dist:
                                valid_pol.append(new_pol)

                    prob = 1.0 / len(valid_pol)
                    for new_pol in valid_pol:
                        if new_p == new_pol:
                            reward = -50
                            next_v = 0
                        else:
                            reward = 10 if new_p == mdp.bank_pos else 0
                            next_v = V.get((new_p, new_pol), 0)

                        expected_val += prob * (reward + gamma * next_v)

                if expected_val > max_val:
                    max_val = expected_val
                    best_a = a

            V[s] = max_val
            policy[s] = best_a
            delta = max(delta, abs(v - max_val))

        if delta < theta:
            break

    return V, policy

def q_learning(mdp, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}

    def get_q(s, a):
        if (s, a) not in Q:
            Q[(s, a)] = 0.0
        return Q[(s, a)]

    for _ in range(episodes):
        s = mdp.reset()
        done = False

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                a = random.choice(mdp.action_space)
            else:
                q_vals = [get_q(s, act) for act in mdp.action_space]
                max_q = max(q_vals)
                best_actions = [act for act, q in zip(mdp.action_space, q_vals) if q == max_q]
                a = random.choice(best_actions)

            next_s, r, done = mdp.step(a)

            # Update Q
            best_next_q = max([get_q(next_s, act) for act in mdp.action_space])
            Q[(s, a)] = get_q(s, a) + alpha * (r + gamma * best_next_q - get_q(s, a))
            s = next_s

    policy = {}
    states = set([k[0] for k in Q.keys()])
    for s in states:
        q_vals = [get_q(s, a) for a in mdp.action_space]
        policy[s] = np.argmax(q_vals)

    return Q, policy

def sarsa(mdp, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}

    def get_q(s, a):
        if (s, a) not in Q:
            Q[(s, a)] = 0.0
        return Q[(s, a)]

    def get_action(s):
        if random.random() < epsilon:
            return random.choice(mdp.action_space)
        q_vals = [get_q(s, act) for act in mdp.action_space]
        max_q = max(q_vals)
        best_actions = [act for act, q in zip(mdp.action_space, q_vals) if q == max_q]
        return random.choice(best_actions)

    for _ in range(episodes):
        s = mdp.reset()
        a = get_action(s)
        done = False

        while not done:
            next_s, r, done = mdp.step(a)
            next_a = get_action(next_s)

            Q[(s, a)] = get_q(s, a) + alpha * (r + gamma * get_q(next_s, next_a) - get_q(s, a))
            s, a = next_s, next_a

    policy = {}
    states = set([k[0] for k in Q.keys()])
    for s in states:
        q_vals = [get_q(s, act) for act in mdp.action_space]
        policy[s] = np.argmax(q_vals)

    return Q, policy