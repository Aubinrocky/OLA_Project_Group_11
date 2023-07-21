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

# Step 3: Learning for joint pricing and advertising

from environment import *
from step1 import *
from step2 import *

C = 'C1'
unit_cost = 1
n_arms_bids = 100  # The seller can choose among 100 possible bids
# No information is known about the advertising curve apriori
bids = np.linspace(0.0, 1.0, n_arms_bids)

sigma = 0.05
# Assumption - no information is known about the prices apriori
prices = [2, 3, 4, 5, 6]
n_arms_prices = len(prices)

n_exp = 3

''' PART 1 ● find the best price  '''
# TS
regret_cum_ts = np.zeros((n_exp, T))
reward_cum_ts = np.zeros((n_exp, T))
regret_instant_ts = np.zeros((n_exp, T))
reward_instant_ts = np.zeros((n_exp, T))
all_pulled_arms = np.zeros((n_exp, T))

for j in range(n_exp):
    # Consider a random bid (for each experiment) as starting point to optimize the price
    random_bid = np.random.choice(bids)
    env = Environment(unit_cost=unit_cost, n_arms_prices=n_arms_prices,
                      n_arms_bids=1, bids=[random_bid], sigma=0)

    # TS
    learner_ts = TS_Learner(C, n_arms=n_arms_prices, arms=prices)
    reward_ts = []
    regret_ts = []

    for t in range(T):
        # TS
        pulled_arm = learner_ts.pull_arm()
        reward = env.round(C=C, bid=random_bid, price=prices[pulled_arm])
        learner_ts.update(C, pulled_arm, reward)
        reward_ts.append(reward)
        reward_instant_ts[j, t] = reward
        all_pulled_arms[j, t] = pulled_arm

    opt = np.max(reward_ts)  # optimal reward from the experiment
    opt_array = np.full_like(reward_ts, opt)

    # TS
    reward_cum_ts[j, :] = np.cumsum(reward_ts)
    regret_cum_ts[j, :] = np.cumsum(opt_array - reward_ts)
    regret_instant_ts[j, :] = opt_array - reward_ts

# Assumption that the optimal price is the one of the arm that has been played the most over all runs
values, counts = np.unique(all_pulled_arms, return_counts=True)
opt_arm = int(values[counts.argmax()])
opt_price = prices[opt_arm]

''' PART 2 ● then optimize the bid '''

# GPTS
regret_cum_gpts = np.zeros((n_exp, T))
reward_cum_gpts = np.zeros((n_exp, T))
regret_instant_gpts = np.zeros((n_exp, T))
reward_instant_gpts = np.zeros((n_exp, T))

# GPUCB
regret_cum_gpucb = np.zeros((n_exp, T))
reward_cum_gpucb = np.zeros((n_exp, T))
regret_instant_gpucb = np.zeros((n_exp, T))
reward_instant_gpucb = np.zeros((n_exp, T))

for j in range(0, n_exp):
    env = Environment(unit_cost=unit_cost, n_arms_prices=1,
                      n_arms_bids=n_arms_bids, bids=bids, sigma=sigma)
    # GPTS
    gpts_learner = GPTS_Learner(C=C, n_arms=n_arms_bids, arms=bids)
    reward_gpts = []
    regret_gpts = []
    # GPUCB
    gpucb_learner = GPUCB_Learner(C=C, n_arms=n_arms_bids, arms=bids)
    reward_gpucb = []
    regret_gpucb = []

    for t in range(0, T):
        # GPTS
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(C=C, bid=bids[pulled_arm], price=opt_price)
        gpts_learner.update(pulled_arm, reward)
        reward_gpts.append(reward)
        reward_instant_gpts[j, t] = reward

        # GPUCB
        pulled_arm = gpucb_learner.pull_arm()
        reward = env.round(C=C, bid=bids[pulled_arm], price=opt_price)
        gpucb_learner.update(pulled_arm, reward)
        reward_gpucb.append(reward)
        reward_instant_gpucb[j, t] = reward

    # optimal reward from the experiment
    opt = max(max(reward_gpucb), max(reward_gpts))
    opt_array = np.full_like(reward_gpucb, opt)

    # GPTS
    reward_cum_gpts[j, :] = np.cumsum(reward_gpts)
    regret_cum_gpts[j, :] = np.cumsum(opt_array - reward_gpts)
    regret_instant_gpts[j, :] = opt_array - reward_gpts

    # GPUCB
    reward_cum_gpucb[j, :] = np.cumsum(reward_gpucb)
    regret_cum_gpucb[j, :] = np.cumsum(opt_array - reward_gpucb)
    regret_instant_gpucb[j, :] = opt_array - reward_gpucb

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values - Price learning')
axs[0, 0].plot(np.mean(regret_cum_ts, axis=0))
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_ts, axis=0))
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_ts, axis=0))
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_ts, axis=0))
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations - Price learning')
axs[0, 0].plot(np.std(regret_cum_ts, axis=0))
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_ts, axis=0))
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_ts, axis=0))
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_ts, axis=0))
axs[1, 1].set_title('Instantaneous reward')

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values')
axs[0, 0].plot(np.mean(regret_cum_gpts, axis=0))
axs[0, 0].plot(np.mean(regret_cum_gpucb, axis=0))
axs[0, 0].legend(['GP-TS', 'GP-UCB'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_gpts, axis=0))
axs[0, 1].plot(np.mean(reward_cum_gpucb, axis=0))
axs[0, 1].legend(['GP-TS', 'GP-UCB'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_gpts, axis=0))
axs[1, 0].plot(np.mean(regret_instant_gpucb, axis=0))
axs[1, 0].legend(['GP-TS', 'GP-UCB'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_gpts, axis=0))
axs[1, 1].plot(np.mean(reward_instant_gpucb, axis=0))
axs[1, 1].legend(['GP-TS', 'GP-UCB'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations')
axs[0, 0].plot(np.std(regret_cum_gpts, axis=0))
axs[0, 0].plot(np.std(regret_cum_gpucb, axis=0))
axs[0, 0].legend(['GP-TS', 'GP-UCB'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_gpts, axis=0))
axs[0, 1].plot(np.std(reward_cum_gpucb, axis=0))
axs[0, 1].legend(['GP-TS', 'GP-UCB'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_gpts, axis=0))
axs[1, 0].plot(np.std(regret_instant_gpucb, axis=0))
axs[1, 0].legend(['GP-TS', 'GP-UCB'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_gpts, axis=0))
axs[1, 1].plot(np.std(reward_instant_gpucb, axis=0))
axs[1, 1].legend(['GP-TS', 'GP-UCB'])
axs[1, 1].set_title('Instantaneous reward')
