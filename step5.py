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

# Step 5: Dealing with non-stationary environments with two abrupt changes

from environment import *
from step1 import *
from step2 import *
from step3 import *
from step4_1 import *


class UCB_SW (Learner):
    def __init__(self, C, sw_size, n_arms, arms):
        super().__init__(C, n_arms, arms)
        self.sw_size = sw_size
        self.valid_rewards = []
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf]*self.n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update_window(self, pull_arm, reward):
        self.valid_rewards.append((pull_arm, reward))

        if len(self.valid_rewards) > self.sw_size:
            self.valid_rewards = self.valid_rewards[1:]

    def update_model(self):
        self.empirical_means = np.zeros(self.n_arms)
        self.confidence = np.array([np.inf]*self.n_arms)
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.t = 0

        for day in self.valid_rewards:
            self.t += 1
            arm_idx = day[0]
            reward = day[1]
            self.rewards_per_arm[arm_idx].append(reward)
            self.collected_rewards = np.append(self.collected_rewards, reward)
            self.empirical_means[arm_idx] = (
                self.empirical_means[arm_idx]*(self.t-1)+reward)/self.t
        # for a in range(self.n_arms):
        #   n_samples = len(self.rewards_per_arm[a])
        #   self.empirical_means[arm_idx] = sum(self.rewards_per_arm[a])/n_samples if n_samples>0 else np.inf
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (
                2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf

    def update(self, pulled_arm, reward):
        self.update_window(pulled_arm, reward)

        self.update_model()

# Change detection (CUSUM algorithm)


class UCB_CD (Learner):
    def __init__(self, C, M, eps, h, n_arms, arms):
        super().__init__(C, n_arms, arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        self.change_detection = ChangeDetection(n_arms, M, eps, h)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm, reward):
        self.t += 1

        if self.change_detection.update(pull_arm, reward):
            self.empirical_means[pull_arm] = reward
            self.rewards_per_arm[pull_arm] = [reward]
        else:
            self.empirical_means[pull_arm] = (
                self.empirical_means[pull_arm]*(self.t-1)+reward)/self.t
            self.rewards_per_arm[pull_arm].append(reward)

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (
                2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf


class SingleArmDetector:

    def __init__(self, M, eps, h):

        # number of samples needed to initialize the reference value
        self.M = M

        # reference value
        self.reference = 0

        # epsilon value, parameter of change detection formula
        self.eps = eps

        # threshold to detect the change
        self.h = h

        # g values that will be computed by the algorithm
        self.g_plus = 0
        self.g_minus = 0

        # number of rounds executed
        self.t = 0

    def update(self, sample):
        self.t += 1

        if self.t <= self.M:
            self.reference += sample/self.M
            return False
        else:
            self.reference = (self.reference*(self.t-1) + sample)/self.t
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps

            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)

            if self.g_minus > self.h or self.g_plus > self.h:

                self.reset()
                return True
            return False

    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
        self.reference = 0


class ChangeDetection:
    def __init__(self, n_arms, M, eps, h):
        self.n_arms = n_arms
        self.detectors = [SingleArmDetector(
            M, eps, h) for arm in range(n_arms)]

    def update(self, arm, sample):
        return self.detectors[arm].update(sample)


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

# UCB_1
reward_cum_ucb = np.zeros((n_exp, T))
regret_cum_ucb = np.zeros((n_exp, T))
reward_instant_ucb = np.zeros((n_exp, T))
regret_instant_ucb = np.zeros((n_exp, T))

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
    # UCB_1
    learner_ucb = UCB(C, n_arms=len(prices[0]), arms=prices)
    reward_ucb = []
    regret_ucb = []

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

        # UCB_1
        pulled_arm = learner_ucb.pull_arm()
        reward = env.round(
            C=C, bid=opt_bids[0], price=prices[season][pulled_arm])
        learner_ucb.update(pulled_arm, reward)
        reward_ucb.append(reward)

        reward_instant_ucb[j, t] = reward

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

    opt = max(np.max(reward_instant_ucb), np.max(reward_instant_ucb_sw), np.max(
        reward_instant_ucb_cd))  # optimal reward from the experiment in season 0
    opt_array = np.full_like(reward_ucb, opt)

    # UCB_1
    reward_cum_ucb[j, :] = np.cumsum(reward_ucb)
    regret_cum_ucb[j, :] = np.cumsum(opt_array - reward_ucb)
    regret_instant_ucb[j, :] = opt_array - reward_ucb

    # UCB_SW
    reward_cum_ucb_sw[j, :] = np.cumsum(reward_ucb_sw)
    regret_cum_ucb_sw[j, :] = np.cumsum(opt_array - reward_ucb_sw)
    regret_instant_ucb_sw[j, :] = opt_array - reward_ucb_sw

    # UCB_CD
    reward_cum_ucb_cd[j, :] = np.cumsum(reward_ucb_cd)
    regret_cum_ucb_cd[j, :] = np.cumsum(opt_array - reward_ucb_cd)
    regret_instant_ucb_cd[j, :] = opt_array - reward_ucb_cd

# Sensitiviy analysis CD

reward_cum_ucb_cd_avg = np.mean(reward_cum_ucb, axis=0)
reward_cum_ucb_cd_last = reward_cum_ucb_cd_avg[-1]


fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values')
axs[0, 0].plot(np.mean(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['UCB', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['UCB',  'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['UCB',  'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['UCB',  'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')

fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations')
axs[0, 0].plot(np.std(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['UCB', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['UCB', 'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['UCB', 'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['UCB', 'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')
