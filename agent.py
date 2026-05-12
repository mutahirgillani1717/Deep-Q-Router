import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class RouterNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(RouterNetwork, self).__init__()
        # Lightweight architecture optimized for CPU inference
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Memory buffer for Experience Replay
        self.memory = deque(maxlen=2000) 
        
        # Hyperparameters
        self.gamma = 0.95    # Discount rate (cares about future rewards)
        self.epsilon = 1.0   # Exploration rate (starts at 100% random)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize PyTorch Model and Optimizer
        self.model = RouterNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Saves experiences to memory for batch training."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action based on exploration vs. exploitation."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_tensor)     # Exploit
        return np.argmax(act_values.numpy()[0])

    def replay(self, batch_size):
        """Trains the neural network using random samples from memory."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                # FIXED: Using pure PyTorch math to prevent type errors
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).clone().detach()
            
            # target is now a standard Python float, perfectly safe to assign
            target_f[0][action] = target
            
            # Gradient Descent step
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
            
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay