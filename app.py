import streamlit as st
import numpy as np
import time
import torch
from env import NetworkEnvironment
from agent import DQNAgent

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep-Q Router Dashboard", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #09090b; color: #10b981; }
    .stButton>button { background-color: #10b981; color: black; font-weight: bold; width: 100%; }
    .grid-box { font-size: 30px; text-align: center; padding: 20px; background-color: #111827; border-radius: 8px; margin: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_one_hot(state_idx):
    state = np.zeros(25)
    state[state_idx] = 1.0
    return state

def str_to_tuple(s):
    return tuple(map(int, s.strip('()').split(', ')))

def map_changed():
    """Callback triggered when the user modifies the grid layout."""
    st.session_state.trained = False
    st.session_state.manual_msg = "⚠️ Map altered! Neural Network memory wiped. Retraining required."
    
    # Update environment with new rules
    env = st.session_state.env
    env.start_node = str_to_tuple(st.session_state.ui_start)
    env.end_node = str_to_tuple(st.session_state.ui_end)
    env.obstacles = [str_to_tuple(o) for o in st.session_state.ui_obs]
    env.reset()
    
    # Reset Agent and UI state
    st.session_state.agent = DQNAgent(state_size=25, action_size=4)
    st.session_state.grid_history = [list(env.start_node)]

# --- SESSION STATE INITIALIZATION ---
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'env' not in st.session_state:
    st.session_state.env = NetworkEnvironment(grid_size=5)
if 'agent' not in st.session_state:
    st.session_state.agent = DQNAgent(state_size=25, action_size=4)
if 'grid_history' not in st.session_state:
    st.session_state.grid_history = []
if 'manual_msg' not in st.session_state:
    st.session_state.manual_msg = None

# Grid coordinate strings for UI dropdowns
all_nodes = [f"({r}, {c})" for r in range(5) for c in range(5)]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚡ Control Panel")
    
    # --- NEW: ENVIRONMENT BUILDER ---
    st.header("🗺️ Environment Builder")
    st.selectbox("🚀 Start Node:", all_nodes, index=0, key="ui_start", on_change=map_changed)
    st.selectbox("🎯 End Node:", all_nodes, index=24, key="ui_end", on_change=map_changed)
    st.multiselect(
        "🟥 Congested Nodes (Obstacles):", 
        all_nodes, 
        default=["(1, 1)", "(2, 2)", "(3, 1)"], 
        key="ui_obs", 
        on_change=map_changed
    )
    
    st.write("---")
    
    # --- TRAINING AND SIMULATION ---
    if st.button("🧠 Train Neural Network (500 Episodes)"):
        with st.spinner("Crunching MDP & Gradients..."):
            env = st.session_state.env
            agent = st.session_state.agent
            
            # Reset environment to user's custom start node
            env.reset()
            
            for e in range(500):
                state_idx = env.reset()
                state = get_one_hot(state_idx)
                for _ in range(50):
                    action = agent.act(state)
                    next_state_idx, reward, done = env.step(action)
                    next_state = get_one_hot(next_state_idx)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done: break
                for _ in range(3):
                    agent.replay(32)
            
            agent.epsilon = 0.0 # Lock into exploitation mode
            st.session_state.trained = True
            st.session_state.grid_history = [list(env.start_node)]
            st.session_state.manual_msg = "✅ Weights Optimized for new map topography!"
        st.rerun()

    if st.button("🚀 Run Auto-Pilot Simulation"):
        if not st.session_state.trained:
            st.error("Please train the network first!")
        else:
            env = st.session_state.env
            agent = st.session_state.agent
            state_idx = env.reset()
            
            st.session_state.grid_history = [list(env.current_pos)]
            state = get_one_hot(state_idx)
            
            for step in range(25): # Added safety limit
                action = agent.act(state)
                next_state_idx, _, done = env.step(action)
                state = get_one_hot(next_state_idx)
                st.session_state.grid_history.append(list(env.current_pos))
                if done: break

    st.write("---")
    if st.button("🔄 Reset Grid Position"):
        st.session_state.env.reset()
        st.session_state.grid_history = [list(st.session_state.env.start_node)]
        st.session_state.manual_msg = None
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("⚡ Deep-Q Router: Interactive RL Agent")
st.write("Build a custom network topology, train the Deep Q-Network, and watch it discover the optimal path.")
st.write("---")

col1, col2 = st.columns([2, 1])

# Determine current location for rendering
env = st.session_state.env
current_loc = st.session_state.grid_history[-1] if st.session_state.grid_history else list(env.start_node)

with col1:
    st.subheader("📡 Live Network Grid")
    st.caption("🚀 = Packet | 🎯 = End Node | 🟥 = Network Congestion | • = Trajectory")
    
    # Display transient messages (like invalidation warnings or win states)
    if st.session_state.manual_msg:
        if "SUCCESS" in st.session_state.manual_msg or "✅" in st.session_state.manual_msg:
            st.success(st.session_state.manual_msg)
        elif "⚠️" in st.session_state.manual_msg:
            st.warning(st.session_state.manual_msg)
        else:
            st.error(st.session_state.manual_msg)
        st.session_state.manual_msg = None # Clear after displaying
    
    # Grid Rendering Logic
    for i in range(5):
        cols = st.columns(5)
        for j in range(5):
            pos = [i, j]
            with cols[j]:
                if pos == current_loc:
                    st.markdown('<div class="grid-box">🚀</div>', unsafe_allow_html=True)
                elif tuple(pos) == env.end_node:
                    st.markdown('<div class="grid-box">🎯</div>', unsafe_allow_html=True)
                elif tuple(pos) in env.obstacles:
                    st.markdown('<div class="grid-box">🟥</div>', unsafe_allow_html=True)
                else:
                    if pos in st.session_state.grid_history:
                        st.markdown('<div class="grid-box" style="color:#047857;">•</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="grid-box">⬜</div>', unsafe_allow_html=True)

with col2:
    st.subheader("🤖 AI Intelligence Query")
    st.write("Select a coordinate to see the AI's internal predictions, or manually drive the packet.")
    
    if st.session_state.trained:
        st.write("**Target Coordinate Probe:**")
        probe_row = st.selectbox("Select Row (Y):", [0, 1, 2, 3, 4], index=current_loc[0])
        probe_col = st.selectbox("Select Column (X):", [0, 1, 2, 3, 4], index=current_loc[1])
        
        probe_idx = probe_row * 5 + probe_col
        state_tensor = torch.FloatTensor(get_one_hot(probe_idx)).unsqueeze(0)
        
        with torch.no_grad():
            q_values = st.session_state.agent.model(state_tensor).numpy()[0]
        
        st.write("---")
        st.write(f"**Predicted Q-Values for Node ({probe_row}, {probe_col}):**")
        st.write(f"⬆️ **UP:** `{q_values[0]:.2f}`")
        st.write(f"➡️ **RIGHT:** `{q_values[1]:.2f}`")
        st.write(f"⬇️ **DOWN:** `{q_values[2]:.2f}`")
        st.write(f"⬅️ **LEFT:** `{q_values[3]:.2f}`")
        
        best_move = ["UP", "RIGHT", "DOWN", "LEFT"][np.argmax(q_values)]
        st.info(f"**Agent Decision:** If placed at Node ({probe_row}, {probe_col}), the AI would move **{best_move}**.")
        
        # --- MANUAL OVERRIDE LOGIC ---
        st.write("---")
        st.write("**🕹️ Manual Override:**")
        if st.button("🚀 Execute Manual Move to Target"):
            if tuple(current_loc) == env.end_node:
                st.warning("Packet has already reached the destination! Reset the grid to play again.")
            else:
                dr = probe_row - current_loc[0]
                dc = probe_col - current_loc[1]
                
                # Validation: The move must be exactly 1 step (Manhattan distance == 1)
                if abs(dr) + abs(dc) == 1:
                    if dr == -1: action = 0   # UP
                    elif dc == 1: action = 1  # RIGHT
                    elif dr == 1: action = 2  # DOWN
                    elif dc == -1: action = 3 # LEFT
                    
                    next_state_idx, reward, done = env.step(action)
                    st.session_state.grid_history.append(list(env.current_pos))
                    
                    if done:
                        st.session_state.manual_msg = "[SUCCESS] Packet routed manually to the target!"
                    elif reward == -10:
                        st.session_state.manual_msg = "[FAILURE] Packet hit network congestion!"
                        
                    st.rerun() 
                    
                elif abs(dr) + abs(dc) == 0:
                    st.warning("The rocket is already at this node.")
                else:
                    st.error(f"❌ Illegal Move! The rocket cannot jump from {current_loc} to [{probe_row}, {probe_col}].")
    else:
        st.warning("Train the agent to unlock internal metric querying and manual overrides.")

st.write("---")
st.caption("Architecture: PyTorch DQN | Experience Replay Enabled | UI: Event-Driven State Management")