import numpy as np
import random
import tkinter as tk

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

    def render(self, canvas):
        canvas.delete("all")
        for r in range(self.size):
            for c in range(self.size):
                color = "white"
                if (r, c) == self.goal:
                    color = "green"
                elif (r, c) == self.state:
                    color = "blue"
                elif (r, c) in self.obstacles:
                    color = "red"
                canvas.create_rectangle(c * 50, r * 50, c * 50 + 50, r * 50 + 50, fill=color, outline="black")

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

# Main GUI application
class App:
    def __init__(self, root, env, q_table):
        self.root = root
        self.env = env
        self.q_table = q_table
        self.canvas = tk.Canvas(root, width=env.size * 50, height=env.size * 50)
        self.canvas.pack()
        self.train_button = tk.Button(root, text="Train Agent", command=self.train_agent)
        self.train_button.pack()
        self.test_button = tk.Button(root, text="Test Agent", command=self.test_agent)
        self.test_button.pack()
        self.env.render(self.canvas)

    def train_agent(self):
        self.q_table = q_learning(self.env, num_episodes=1000, alpha=0.1, gamma=0.6, epsilon=0.1)
        self.env.render(self.canvas)

    def test_agent(self):
        state = self.env.reset()
        self.env.render(self.canvas)
        self.root.after(500, self.perform_action)

    def perform_action(self):
        action = np.argmax(self.q_table[self.env.state[0], self.env.state[1]])
        state, _, done = self.env.step(action)
        self.env.render(self.canvas)
        if not done:
            self.root.after(500, self.perform_action)

# Main code
if __name__ == "__main__":
    env = Environment(size=5)
    q_table = q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.6, epsilon=0.1)

    root = tk.Tk()
    app = App(root, env, q_table)
    root.mainloop()
