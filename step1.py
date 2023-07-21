# Step 1: Learning for pricing

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

from environment import *


class UCB (Learner):
    def __init__(self, C, n_arms, arms):
        super().__init__(C, n_arms, arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm, reward):
        self.t += 1
        self.empirical_means[pull_arm] = (
            self.empirical_means[pull_arm]*(self.t-1)+reward)/self.t

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (
                2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)


class TS_Learner (Learner):
    def __init__(self, C, n_arms, arms):
        super().__init__(C, n_arms, arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(
            self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    ''' Modified update function as the reward is neither strictly equal to 0 or 1'''

    def update(self, C, pulled_arm, reward):
        # Values obtain from previous trials (empirical data)
        self.opt_empirical = [48, 62, 42]
        self.t += 1
        self.update_observations(pulled_arm, reward)

        if C == 'C1':
            opt = self.opt_empirical[0]

        elif C == 'C2':
            opt = self.opt_empirical[1]

        else:
            opt = self.opt_empirical[2]

        if reward >= 0:
            # 1st parameter: how many success we have had -> add something proportionnal to the succes of the arm
            self.beta_parameters[pulled_arm,
                                 0] = self.beta_parameters[pulled_arm, 0] + reward/opt
            # 2nd parameter: opposite
            self.beta_parameters[pulled_arm,
                                 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward/opt
        else:
            # If the reward < 0, it represents a complete failure
            self.beta_parameters[pulled_arm,
                                 1] = self.beta_parameters[pulled_arm, 1] + 1


C = 'C1'  # User class
# Optimal bids (part about advertising is supposed to be known)
opt_bids = [0.5, 0.3, 0.7]
unit_cost = 1  # Assumption
prices = [2, 3, 4, 5, 6]  # Assumption

n_exp = 20
# UCB_1
reward_cum_ucb = np.zeros((n_exp, T))
regret_cum_ucb = np.zeros((n_exp, T))
reward_instant_ucb = np.zeros((n_exp, T))
regret_instant_ucb = np.zeros((n_exp, T))
# TS
reward_cum_ts = np.zeros((n_exp, T))
regret_cum_ts = np.zeros((n_exp, T))
reward_instant_ts = np.zeros((n_exp, T))
regret_instant_ts = np.zeros((n_exp, T))

for j in range(n_exp):
    env = Environment(unit_cost=unit_cost, n_arms_prices=len(
        prices), n_arms_bids=1, bids=[opt_bids[0]], sigma=0)
    # UCB_1
    learner_ucb = UCB(C, n_arms=len(prices), arms=prices)
    reward_ucb = []
    regret_ucb = []
    # TS
    learner_ts = TS_Learner(C, n_arms=len(prices), arms=prices)
    reward_ts = []
    regret_ts = []

    for t in range(T):
        # UCB_1
        pulled_arm = learner_ucb.pull_arm()
        reward = env.round(C=C, bid=opt_bids[0], price=prices[pulled_arm])
        learner_ucb.update(pulled_arm, reward)
        reward_ucb.append(reward)
        reward_instant_ucb[j, t] = reward

        # TS
        pulled_arm = learner_ts.pull_arm()
        reward = env.round(C=C, bid=opt_bids[0], price=prices[pulled_arm])
        learner_ts.update(C, pulled_arm, reward)
        reward_ts.append(reward)
        reward_instant_ts[j, t] = reward

    # Optimal reward is set as the maximum reward from the experiment
    opt = max(max(reward_ucb), max(reward_ts))
    opt_array = np.full_like(reward_ucb, opt)

    # UCB_1
    reward_cum_ucb[j, :] = np.cumsum(reward_ucb)
    regret_cum_ucb[j, :] = np.cumsum(opt_array - reward_ucb)
    regret_instant_ucb[j, :] = opt_array - reward_ucb
    # TS
    reward_cum_ts[j, :] = np.cumsum(reward_ts)
    regret_cum_ts[j, :] = np.cumsum(opt_array - reward_ts)
    regret_instant_ts[j, :] = opt_array - reward_ts

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values')
axs[0, 0].plot(np.mean(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ts, axis=0))
axs[0, 0].legend(['UCB', 'TS'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ts, axis=0))
axs[0, 1].legend(['UCB', 'TS'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ts, axis=0))
axs[1, 0].legend(['UCB', 'TS'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ts, axis=0))
axs[1, 1].legend(['UCB', 'TS'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations')
axs[0, 0].plot(np.std(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.std(regret_cum_ts, axis=0))
axs[0, 0].legend(['UCB', 'TS'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.std(reward_cum_ts, axis=0))
axs[0, 1].legend(['UCB', 'TS'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.std(regret_instant_ts, axis=0))
axs[1, 0].legend(['UCB', 'TS'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.std(reward_instant_ts, axis=0))
axs[1, 1].legend(['UCB', 'TS'])
axs[1, 1].set_title('Instantaneous reward')
