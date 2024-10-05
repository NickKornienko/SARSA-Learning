#!/usr/bin/env python
"""
sarsa_learner.py
An easy-to-follow script to train, test and evaluate a SARSA agent on the Mountain Car
problem using the Gymnasium library.
"""

import gymnasium as gym
import numpy as np

MAX_NUM_EPISODES = 100000
# Specific to MountainCar. May change with other environments.
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to discretize each observation dimension
# Decay epsilon over the episodes
EPSILON_DECAY = (1.0 - EPSILON_MIN) / MAX_NUM_EPISODES


class SARSA_Learner:
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins for discretization
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins, self.obs_bins,
                          self.action_shape))  # Q-table
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        """Discretizes the continuous observation space."""
        discrete_obs = (obs - self.obs_low) / self.bin_width
        return tuple(np.clip(discrete_obs.astype(int), 0, self.obs_bins - 1))

    def get_action(self, obs):
        """Epsilon-greedy policy for action selection."""
        discretized_obs = self.discretize(obs)
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:
            return np.random.choice(self.action_shape)

    def learn(self, obs, action, reward, next_obs, next_action):
        """SARSA update rule."""
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * \
            self.Q[discretized_next_obs][next_action]
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error

        # Epsilon decay
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY


def train(agent, env):
    """Train the SARSA agent."""
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs, _ = env.reset()  # Gymnasium reset returns observation and info
        action = agent.get_action(obs)
        total_reward = 0.0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Check for both terminated and truncated
            next_action = agent.get_action(next_obs)

            agent.learn(obs, action, reward, next_obs, next_action)

            obs = next_obs
            action = next_action
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward

        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Best: {
                  best_reward}, Epsilon: {agent.epsilon}")

    # Return the trained policy
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    """Test the agent's learned policy."""
    obs, _ = env.reset()  # Gymnasium reset returns observation and info
    total_reward = 0.0
    done = False

    while not done:
        action = policy[agent.discretize(obs)]
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Check for both terminated and truncated
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    # Initialize environment without rendering for training
    env = gym.make('MountainCar-v0', render_mode=None)
    agent = SARSA_Learner(env)

    # Train the agent
    learned_policy = train(agent, env)

    # Record video of the final episode (last test step only)
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    # Trigger for final episode only
    env = gym.wrappers.RecordVideo(
        env, "./gym_monitor_output", episode_trigger=lambda x: x == 0)

    # Perform one test episode to record
    test(agent, env, learned_policy)

    env.close()
