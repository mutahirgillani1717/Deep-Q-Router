from env import NetworkEnvironment
from agent import DQNAgent
import numpy as np

# Hyperparameters
EPISODES = 300
GRID_SIZE = 5
BATCH_SIZE = 32

def get_one_hot_state(state_idx, size):
    """Converts the integer grid position into a binary array for the Neural Net."""
    state = np.zeros(size)
    state[state_idx] = 1.0
    return state

if __name__ == "__main__":
    print("[*] Initiating Deep-Q Training Sequence...")
    
    env = NetworkEnvironment(grid_size=GRID_SIZE)
    state_size = env.state_space
    action_size = env.action_space
    
    agent = DQNAgent(state_size, action_size)

    for e in range(EPISODES):
        state_idx = env.reset()
        state = get_one_hot_state(state_idx, state_size)
        
        # Max 50 steps per episode to prevent infinite loops
        for time_step in range(50): 
            action = agent.act(state)
            next_state_idx, reward, done = env.step(action)
            next_state = get_one_hot_state(next_state_idx, state_size)
            
            # Save the experience to memory
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode: {e+1}/{EPISODES} | Route found in {time_step} steps | Epsilon (Exploration): {agent.epsilon:.2f}")
                break
                
            # Train the network using past memories
            agent.replay(BATCH_SIZE)
            
        # If the packet got lost/didn't find the end in 50 steps
        if not done:
             print(f"Episode: {e+1}/{EPISODES} | Packet Lost (Time Out) | Epsilon: {agent.epsilon:.2f}")

    print("[+] Training Complete. Neural Weights Optimized.")