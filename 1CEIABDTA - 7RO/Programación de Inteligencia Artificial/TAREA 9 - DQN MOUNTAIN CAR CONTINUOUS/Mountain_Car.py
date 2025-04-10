import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('MountainCarContinuous-v0')

# Parameters
episodes = 1000
timesteps = 20
learning_rate = 0.01
discount_factor = 2

# Discretize the state space
state_bins = [20, 20]  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(bounds[0], bounds[1], num=bins) for bounds, bins in zip(state_bounds, state_bins)]

def discretize_state(state):
    """Discretize a continuous state into a discrete index."""
    state_idx = []
    for i, value in enumerate(state):
        state_idx.append(np.digitize(value, state_bins[i]) - 1)  # Bin index starts from 0
    return tuple(state_idx)

# Initialize Q-table
q_table = np.zeros([len(bins) + 1 for bins in state_bins] + [1])  # Action space is continuous, so this is a placeholder

# Helper function to choose action
def choose_action(state):
    return env.action_space.sample()  # Random action for now

# Training loop
rewards = []
for episode in range(episodes):
    state = discretize_state(env.reset()[0])  # Discretize the initial state
    print(f"pos: {float(state[0])}, vel: {float(state[1])}")
    total_reward = 0
    for t in range(timesteps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)  # Discretize the next state
        total_reward += reward

        # Update the Q-table
        q_table[state] = q_table[state] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state]
        )

        state = next_state
        if terminated or truncated:
            break
    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Plotting the rewards
plt.figure(figsize=(12, 6))

# Raw rewards
plt.plot(rewards, label='Raw Rewards', alpha=0.5)

# Moving average of rewards
window_size = 100
moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Average', color='orange')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Diagram: Rewards over Episodes')
plt.legend()
plt.show()

# Close the environment
env.close()