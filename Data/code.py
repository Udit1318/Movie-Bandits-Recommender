"""
bandit_ucb.py

Placeholder for Multi-Armed Bandit algorithms (UCB, Epsilon-Greedy)
for movie recommendation system.

Author: Udit
"""

import numpy as np

class UCBBandit:
    """
    Upper Confidence Bound (UCB) Bandit implementation
    """

    def __init__(self, n_arms):
        """
        Initialize the bandit
        :param n_arms: Number of arms (movies)
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)   # number of times each arm was chosen
        self.values = np.zeros(n_arms)   # average reward of each arm

    def select_arm(self):
        """
        Select the arm with the highest UCB value
        """
        ucb_values = self.values + np.sqrt(2 * np.log(np.sum(self.counts) + 1) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """
        Update the estimated value of an arm
        :param arm: arm index
        :param reward: observed reward (0 or 1)
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

if __name__ == "__main__":
    # Demo: initialize 5-arm bandit and simulate random rewards
    bandit = UCBBandit(n_arms=5)
    for i in range(10):
        arm = bandit.select_arm()
        reward = np.random.choice([0, 1])
        bandit.update(arm, reward)
    print("Dummy run complete. Arm values:", bandit.values)
