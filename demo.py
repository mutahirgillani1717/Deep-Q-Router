from env import NetworkEnvironment
from agent import DQNAgent
import numpy as np
import time
import os

EPISODES = 500
GRID_SIZE = 5
BATCH_SIZE = 32

def get_one_hot_state(state_idx, size):
    state = np.zeros(size)
    state[state_idx] = 1.0
    return state

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    clear_console()
    print("[*] Initializing CPU-Optimized Deep-Q Router...")
    
    env = NetworkEnvironment(grid_size=GRID_SIZE)
    state_size = env.state_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    print("[*] Training Neural Network. This will be fast...\n")
    
    # --- CPU-OPTIMIZED TRAINING PHASE ---
    for e in range(EPISODES):
        state_idx = env.reset()
        state = get_one_hot_state(state_idx, state_size)
        
        # Progress Indicator
        if e % 100 == 0 or e == EPISODES - 1:
            print(f"    -> Crunching episode {e}/{EPISODES} | Epsilon: {agent.epsilon:.2f}")

        # 1. Rollout Phase: Just gather experience (No math here)
        for time_step in range(50): 
            action = agent.act(state)
            next_state_idx, reward, done = env.step(action)
            next_state = get_one_hot_state(next_state_idx, state_size)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done: break
            
        # 2. Optimization Phase: Train in bursts at the end of the episode
        # Reduces CPU backprop passes from 15,000+ down to exactly 1,500
        for _ in range(3): 
            agent.replay(BATCH_SIZE)

    # --- LIVE VISUALIZATION PHASE ---
    print("\n[+] Weights Optimized. Initiating Live Routing Demo in 2 seconds...")
    time.sleep(2)
    
    agent.epsilon = 0.0 
    state_idx = env.reset()
    state = get_one_hot_state(state_idx, state_size)
    
    steps_taken = 0
    for step in range(20): 
        clear_console()
        print(f"--- LIVE ROUTING (Step {steps_taken}) ---")
        env.render()
        
        action = agent.act(state)
        next_state_idx, reward, done = env.step(action)
        next_state = get_one_hot_state(next_state_idx, state_size)
        state = next_state
        steps_taken += 1
        
        if done:
            clear_console()
            print(f"--- LIVE ROUTING (Step {steps_taken}) ---")
            env.render()
            print(f"\n[SUCCESS] Packet routed securely in {steps_taken} steps!")
            break
        elif reward == -10:
            clear_console()
            print(f"--- LIVE ROUTING (Step {steps_taken}) ---")
            env.render()
            print(f"\n[FAILURE] Packet hit network congestion!")
            break
            
        time.sleep(0.4)