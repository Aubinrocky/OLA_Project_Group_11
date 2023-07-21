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
from step6_1 import *

# Part 2


C = 'C1'  # user class
# Optimal bids (part about advertising is supposed to be known)
opt_bids = [0.5, 0.3, 0.7]

unit_cost = 1  # Assumption

# Three different phases with different pricing curves
prices_1 = [7, 8, 11, 13, 14]  # Phase 1
prices_2 = [4, 6, 8, 12, 13]  # Phase 2
prices_3 = [2, 5, 6, 9, 10]  # Phase 3
prices_4 = [9, 12, 13, 15, 16]  # Phase 4
prices_5 = [6, 8, 9, 11, 12]  # Phase 5

# Store the prices for each phase in a list
prices = [prices_1, prices_2, prices_3, prices_4, prices_5]

optimal_prices = [10, 12, 15, 17, 16]

n_exp = 100

# EXP3
eta = 0.5
reward_cum_exp3 = np.zeros((n_exp, T))
regret_cum_exp3 = np.zeros((n_exp, T))
reward_instant_exp3 = np.zeros((n_exp, T))
regret_instant_exp3 = np.zeros((n_exp, T))

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
    # EXP3
    learner_exp3 = EXP3(C, n_arms=len(prices[0]), arms=prices, eta=eta)
    reward_exp3 = []
    regret_exp3 = []

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

        # Determine the current phase based on phase_duration
        season = (t // 14) % 5

        # EXP3
        pulled_arm = learner_exp3.pull_arm()
        reward = env.round(
            C=C, bid=opt_bids[0], price=prices[season][pulled_arm])
        learner_exp3.update(pulled_arm, reward)
        reward_exp3.append(reward)

        reward_instant_exp3[j, t] = reward

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

    opt = max(np.max(reward_instant_exp3), np.max(reward_instant_ucb), np.max(reward_instant_ucb_sw), np.max(
        reward_instant_ucb_cd))  # optimal reward from the experiment in season 0
    opt_array = np.full_like(reward_exp3, opt)

    # EXP3
    reward_cum_exp3[j, :] = np.cumsum(reward_exp3)
    regret_cum_exp3[j, :] = np.cumsum(opt_array - reward_exp3)
    regret_instant_exp3[j, :] = opt_array - reward_exp3

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

fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values')
axs[0, 0].plot(np.mean(regret_cum_exp3, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_exp3, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_exp3, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_exp3, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')

fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations')
axs[0, 0].plot(np.std(regret_cum_exp3, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_sw, axis=0))
axs[0, 0].plot(np.std(regret_cum_ucb_cd, axis=0))
axs[0, 0].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_exp3, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_sw, axis=0))
axs[0, 1].plot(np.std(reward_cum_ucb_cd, axis=0))
axs[0, 1].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_exp3, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_sw, axis=0))
axs[1, 0].plot(np.std(regret_instant_ucb_cd, axis=0))
axs[1, 0].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_exp3, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_sw, axis=0))
axs[1, 1].plot(np.std(reward_instant_ucb_cd, axis=0))
axs[1, 1].legend(['EXP3', 'UCB', 'SW', 'CD'])
axs[1, 1].set_title('Instantaneous reward')
