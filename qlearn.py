import numpy as np
import random

# Define the environment
class Environment:
    def __init__(self, size):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2)]
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        row, col = self.state
        if action == 0:  # up
            row -= 1
        elif action == 1:  # down
            row += 1
        elif action == 2:  # left
            col -= 1
        elif action == 3:  # right
            col += 1

        next_state = (max(0, min(row, self.size-1)), max(0, min(col, self.size-1)))

        if next_state in self.obstacles:
            next_state = self.state
        
        reward = -1
        done = False
        if next_state == self.goal:
            reward = 100
            done = True
        
        self.state = next_state
        return next_state, reward, done

    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.goal] = 2
        for ob in self.obstacles:
            grid[ob] = -1
        grid[self.state] = 1
        print(grid)

# Q-learning algorithm
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.size, env.size, 4))

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2, 3])  # Explore action space
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit learned values
            
            next_state, reward, done = env.step(action)
            
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state[0], state[1], action] = new_value
            
            state = next_state

    return q_table

# Main code
env = Environment(size=5)
q_table = q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.6, epsilon=0.1)

# Test the learned policy
state = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(q_table[state[0], state[1]])
    state, _, done = env.step(action)
    env.render()
