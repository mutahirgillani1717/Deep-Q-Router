import numpy as np

class NetworkEnvironment:
    def __init__(self, grid_size=5):
        """Initializes the server grid environment."""
        self.grid_size = grid_size
        self.state_space = self.grid_size * self.grid_size
        self.action_space = 4  # Up, Down, Left, Right
        
        # Define grid coordinates
        self.start_node = (0, 0)
        self.end_node = (grid_size - 1, grid_size - 1)
        
        # Define obstacles (Simulating network congestion/firewalls)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        self.reset()

    def reset(self):
        """Resets the packet to the start node."""
        self.current_pos = list(self.start_node)
        return self._get_state()

    def _get_state(self):
        """Converts the 2D coordinate into a 1D state index."""
        return self.current_pos[0] * self.grid_size + self.current_pos[1]

    def step(self, action):
        """
        Takes an action and returns the new state, reward, and if it's done.
        Actions: 0: Up, 1: Right, 2: Down, 3: Left
        """
        x, y = self.current_pos

        # Calculate new position
        if action == 0 and x > 0:            # Up
            x -= 1
        elif action == 1 and y < self.grid_size - 1: # Right
            y += 1
        elif action == 2 and x < self.grid_size - 1: # Down
            x += 1
        elif action == 3 and y > 0:            # Left
            y -= 1

        self.current_pos = [x, y]
        new_state = self._get_state()

        # Check Rewards and Penalties
        if tuple(self.current_pos) == self.end_node:
            reward = 100  # Packet successfully routed!
            done = True
        elif tuple(self.current_pos) in self.obstacles:
            reward = -10  # Hit a congested node
            done = False
        else:
            reward = -1   # Standard step penalty (encourages the fastest route)
            done = False

        return new_state, reward, done

    def render(self):
        """A simple terminal visualization of the network grid."""
        print("-" * (self.grid_size * 4 + 1))
        for i in range(self.grid_size):
            row = "|"
            for j in range(self.grid_size):
                pos = (i, j)
                if list(pos) == self.current_pos:
                    row += " P |" # Packet
                elif pos == self.end_node:
                    row += " E |" # End Node
                elif pos in self.obstacles:
                    row += " X |" # Congestion
                else:
                    row += "   |" # Open path
            print(row)
            print("-" * (self.grid_size * 4 + 1))

if __name__ == "__main__":
    # Quick test to make sure the environment works
    print("[*] Initializing Network Routing Environment...")
    env = NetworkEnvironment(grid_size=5)
    env.render()