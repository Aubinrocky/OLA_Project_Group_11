import pandas as pd
import math
import seaborn as sns
import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK

# Step 6: Dealing with non-stationary environments with many abrupt changes

from environment import *
from step1 import *
from step2 import *
from step3 import *
from step4_1 import *
from step5 import *

# Part 1


class EXP3(Learner):
    def __init__(self, C, n_arms, eta, arms):
        super().__init__(C, n_arms, arms)
        self.eta = eta
        self.weights = np.ones(n_arms)
        self.pulled_arms = []
        self.collected_rewards = []
        self.t = 1

    def update_observations(self, arm_idx, reward):
        self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)

    def update_weights(self):
        estimated_rewards = np.array(
            self.collected_rewards) / self.weights[self.pulled_arms]
        weight_update = np.exp((self.eta / self.n_arms) * estimated_rewards)
        self.weights[self.pulled_arms] *= weight_update

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_weights()

    def pull_arm(self):
        # Calculate the probability distribution over arms (pricing options)
        probabilities = (1 - self.eta) * (self.weights /
                                          np.sum(self.weights)) + self.eta / self.n_arms

        # Choose an arm (pricing option)
        chosen_arm = np.random.choice(range(self.n_arms), p=probabilities)
        return chosen_arm


C = 'C1'  # user class
# Optimal bids (part about advertising is supposed to be known)
opt_bids = [0.5, 0.3, 0.7]

unit_cost = 1  # Assumption

# Three different phases with different pricing curves
# We assume the three curves
prices_1 = [7, 8, 11, 13, 14]  # Regular Pricing Phase
prices_2 = [4, 6, 8, 12, 13]  # Summer Pricing Phase
prices_3 = [2, 5, 6, 9, 10]  # Promotion Pricing Phase

prices = [prices_1, prices_2, prices_3]

n_exp = 100

# EXP3
eta = 0.5
reward_cum_exp3 = np.zeros((n_exp, T))
regret_cum_exp3 = np.zeros((n_exp, T))
reward_instant_exp3 = np.zeros((n_exp, T))
regret_instant_exp3 = np.zeros((n_exp, T))

# UCB_SW
window_size = 70

reward_cum_ucb_sw = np.zeros((n_exp, T))
regret_cum_ucb_sw = np.zeros((n_exp, T))
reward_instant_ucb_sw = np.zeros((n_exp, T))
regret_instant_ucb_sw = np.zeros((n_exp, T))

# UCB_CD
M = 6
eps = 0.3
h = 3.5

reward_cum_ucb_cd = np.zeros((n_exp, T))
regret_cum_ucb_cd = np.zeros((n_exp, T))
reward_instant_ucb_cd = np.zeros((n_exp, T))
regret_instant_ucb_cd = np.zeros((n_exp, T))


for j in range(n_exp):
    env = Environment(unit_cost=unit_cost, n_arms_prices=len(
        prices[0]), n_arms_bids=1, bids=[opt_bids[0]], sigma=0)
    # EXP3
    learner_exp3 = EXP3(C, n_arms=len(prices[0]), arms=prices, eta=eta)
    reward_exp3 = []
    regret_exp3 = []

    # UCB_SW
    learner_ucb_sw = UCB_SW(C, window_size, n_arms=len(prices[0]), arms=prices)
    reward_ucb_sw = []
    regret_ucb_sw = []

    # UCB_CD
    learner_ucb_cd = UCB_CD(C, M, eps, h, n_arms=len(prices[0]), arms=prices)
    reward_ucb_cd = []
    regret_ucb_cd = []

    for t in range(T):

        if t < T//3:
            season = 0
        elif t < (T*2)//3:
            season = 1
        else:
            season = 2

    # EXP3
        pulled_arm = learner_exp3.pull_arm()
        reward = env.round(
            C=C, bid=opt_bids[0], price=prices[season][pulled_arm])
        learner_exp3.update(pulled_arm, reward)
        reward_exp3.append(reward)

        reward_instant_exp3[j, t] = reward

        # UCB_SW
        pulled_arm = learner_ucb_sw.pull_arm()
        reward = env.round(
            C=C, bid=opt_bids[0], price=prices[season][pulled_arm])
        learner_ucb_sw.update(pulled_arm, reward)
        reward_ucb_sw.append(reward)

        reward_instant_ucb_sw[j, t] = reward

        # UCB_CD
        pulled_arm = learner_ucb_cd.pull_arm()
        reward = env.round(
            C=C, bid=opt_bids[0], price=prices[season][pulled_arm])
        learner_ucb_cd.update(pulled_arm, reward)
        reward_ucb_cd.append(reward)

        reward_instant_ucb_cd[j, t] = reward

    opt = max(np.max(reward_instant_exp3), np.max(reward_instant_ucb_sw), np.max(
        reward_instant_ucb_cd))  # optimal reward from the experiment in season 0
    opt_array = np.full_like(reward_exp3, opt)

    # EXP3
    reward_cum_exp3[j, :] = np.cumsum(reward_exp3)
    regret_cum_exp3[j, :] = np.cumsum(opt_array - reward_exp3)
    regret_instant_exp3[j, :] = opt_array - reward_exp3

    # UCB_SW
    reward_cum_ucb_sw[j, :] = np.cumsum(reward_ucb_sw)
    regret_cum_ucb_sw[j, :] = np.cumsum(opt_array - reward_ucb_sw)
    regret_instant_ucb_sw[j, :] = opt_array - reward_ucb_sw

    # UCB_CD
    reward_cum_ucb_cd[j, :] = np.cumsum(reward_ucb_cd)
    regret_cum_ucb_cd[j, :] = np.cumsum(opt_array - reward_ucb_cd)
    regret_instant_ucb_cd[j, :] = opt_array - reward_ucb_cd

fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values')
axs[0, 0].plot(np.mean(regret_cum_exp3, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['EXP3', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_exp3, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['EXP3',  'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_exp3, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['EXP3',  'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_exp3, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['EXP3',  'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')

fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations')
axs[0, 0].plot(np.std(regret_cum_exp3, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['EXP3', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_exp3, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['EXP3', 'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_exp3, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['EXP3', 'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_exp3, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['EXP3', 'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')
