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

# Step 4: Contexts and their generation

from environment import *
from step1 import *
from step2 import *
from step3 import*

''' Scenario 1 - Contexts are known beforehand '''
# Results per class of user

C = ['C1', 'C2', 'C3']
unit_cost = 1
n_arms_bids = 100  # The seller can choose among 100 possible bids
# No information is known about the advertising curve apriori
bids = np.linspace(0.0, 1.0, n_arms_bids)

sigma = 0.05
# Assumption - no information is known about the prices apriori
prices = [2, 3, 4, 5, 6]
n_arms_prices = len(prices)

n_exp = 3

''' PART 1 ● for every single class find the best price, independently from the other classes '''

# TS
regret_cum_ts_C1 = [[] for _ in range(n_exp)]
reward_cum_ts_C1 = [[] for _ in range(n_exp)]
regret_instant_ts_C1 = [[] for _ in range(n_exp)]
reward_instant_ts_C1 = [[] for _ in range(n_exp)]
all_pulled_arms_C1 = []

regret_cum_ts_C2 = [[] for _ in range(n_exp)]
reward_cum_ts_C2 = [[] for _ in range(n_exp)]
regret_instant_ts_C2 = [[] for _ in range(n_exp)]
reward_instant_ts_C2 = [[] for _ in range(n_exp)]
all_pulled_arms_C2 = []

regret_cum_ts_C3 = [[] for _ in range(n_exp)]
reward_cum_ts_C3 = [[] for _ in range(n_exp)]
regret_instant_ts_C3 = [[] for _ in range(n_exp)]
reward_instant_ts_C3 = [[] for _ in range(n_exp)]
all_pulled_arms_C3 = []

for j in range(n_exp):
    # Consider a random bid (for each experiment) as starting point to optimize the price
    random_bid = np.random.choice(bids)
    env = Environment(unit_cost=unit_cost, n_arms_prices=n_arms_prices,
                      n_arms_bids=1, bids=[random_bid], sigma=0)

    # TS - Initialize one learner for each context
    learner_ts_C1 = TS_Learner(C[0], n_arms=n_arms_prices, arms=prices)
    reward_ts_C1 = []
    regret_ts_C1 = []

    learner_ts_C2 = TS_Learner(C[1], n_arms=n_arms_prices, arms=prices)
    reward_ts_C2 = []
    regret_ts_C2 = []

    learner_ts_C3 = TS_Learner(C[2], n_arms=n_arms_prices, arms=prices)
    reward_ts_C3 = []
    regret_ts_C3 = []

    for t in range(T):
        # Randomly select the context (ie.e., ['C1', 'C2', 'C3'] to which the user belongs) as it is known a priori
        context = np.random.choice(C)
        # TS - Implement the learner
        if context == C[0]:
            pulled_arm = learner_ts_C1.pull_arm()
        elif context == C[1]:
            pulled_arm = learner_ts_C2.pull_arm()
        else:
            pulled_arm = learner_ts_C3.pull_arm()

        reward = env.round(C=context, bid=random_bid, price=prices[pulled_arm])

        if context == C[0]:
            learner_ts_C1.update(context, pulled_arm, reward)
            reward_ts_C1.append(reward)
            reward_instant_ts_C1[j].append(reward)
            all_pulled_arms_C1.append(pulled_arm)

        elif context == C[1]:
            learner_ts_C2.update(context, pulled_arm, reward)
            reward_ts_C2.append(reward)
            reward_instant_ts_C2[j].append(reward)
            all_pulled_arms_C2.append(pulled_arm)

        else:
            learner_ts_C3.update(context, pulled_arm, reward)
            reward_ts_C3.append(reward)
            reward_instant_ts_C3[j].append(reward)
            all_pulled_arms_C3.append(pulled_arm)

    opt_C1 = max(reward_ts_C1)
    opt_array_C1 = np.full_like(reward_ts_C1, opt_C1)

    opt_C2 = max(reward_ts_C2)
    opt_array_C2 = np.full_like(reward_ts_C2, opt_C2)

    opt_C3 = max(reward_ts_C3)
    opt_array_C3 = np.full_like(reward_ts_C3, opt_C3)

    # TS - Cumulative KPIs
    reward_cum_ts_C1[j] = np.cumsum(reward_ts_C1)
    regret_cum_ts_C1[j] = np.cumsum(opt_array_C1 - reward_ts_C1)
    regret_instant_ts_C1[j] = opt_array_C1 - reward_ts_C1

    reward_cum_ts_C2[j] = np.cumsum(reward_ts_C2)
    regret_cum_ts_C2[j] = np.cumsum(opt_array_C2 - reward_ts_C2)
    regret_instant_ts_C2[j] = opt_array_C2 - reward_ts_C2

    reward_cum_ts_C3[j] = np.cumsum(reward_ts_C3)
    regret_cum_ts_C3[j] = np.cumsum(opt_array_C3 - reward_ts_C3)
    regret_instant_ts_C3[j] = opt_array_C3 - reward_ts_C3

# Result: optimal price for each context
# Assumption that the optimal price is the one of the arm that has been played the most over all runs
values, counts = np.unique(all_pulled_arms_C1, return_counts=True)
opt_arm_C1 = int(values[counts.argmax()])
opt_price_C1 = prices[opt_arm_C1]

values, counts = np.unique(all_pulled_arms_C2, return_counts=True)
opt_arm_C2 = int(values[counts.argmax()])
opt_price_C2 = prices[opt_arm_C2]

values, counts = np.unique(all_pulled_arms_C3, return_counts=True)
opt_arm_C3 = int(values[counts.argmax()])
opt_price_C3 = prices[opt_arm_C3]


''' PART 2 ● then optimize the bid for each class independently from the other classes '''

# GPTS
regret_cum_gpts_C1 = [[] for _ in range(n_exp)]
reward_cum_gpts_C1 = [[] for _ in range(n_exp)]
regret_instant_gpts_C1 = [[] for _ in range(n_exp)]
reward_instant_gpts_C1 = [[] for _ in range(n_exp)]

regret_cum_gpts_C2 = [[] for _ in range(n_exp)]
reward_cum_gpts_C2 = [[] for _ in range(n_exp)]
regret_instant_gpts_C2 = [[] for _ in range(n_exp)]
reward_instant_gpts_C2 = [[] for _ in range(n_exp)]

regret_cum_gpts_C3 = [[] for _ in range(n_exp)]
reward_cum_gpts_C3 = [[] for _ in range(n_exp)]
regret_instant_gpts_C3 = [[] for _ in range(n_exp)]
reward_instant_gpts_C3 = [[] for _ in range(n_exp)]

# GPUCB
regret_cum_gpucb_C1 = [[] for _ in range(n_exp)]
reward_cum_gpucb_C1 = [[] for _ in range(n_exp)]
regret_instant_gpucb_C1 = [[] for _ in range(n_exp)]
reward_instant_gpucb_C1 = [[] for _ in range(n_exp)]

regret_cum_gpucb_C2 = [[] for _ in range(n_exp)]
reward_cum_gpucb_C2 = [[] for _ in range(n_exp)]
regret_instant_gpucb_C2 = [[] for _ in range(n_exp)]
reward_instant_gpucb_C2 = [[] for _ in range(n_exp)]

regret_cum_gpucb_C3 = [[] for _ in range(n_exp)]
reward_cum_gpucb_C3 = [[] for _ in range(n_exp)]
regret_instant_gpucb_C3 = [[] for _ in range(n_exp)]
reward_instant_gpucb_C3 = [[] for _ in range(n_exp)]


for j in range(0, n_exp):
    env = Environment(unit_cost=unit_cost, n_arms_prices=1,
                      n_arms_bids=n_arms_bids, bids=bids, sigma=sigma)
    # GPTS
    gpts_learner_C1 = GPTS_Learner(C=C[0], n_arms=n_arms_bids, arms=bids)
    reward_gpts_C1 = []
    regret_gpts_C1 = []

    gpts_learner_C2 = GPTS_Learner(C=C[1], n_arms=n_arms_bids, arms=bids)
    reward_gpts_C2 = []
    regret_gpts_C2 = []

    gpts_learner_C3 = GPTS_Learner(C=C[2], n_arms=n_arms_bids, arms=bids)
    reward_gpts_C3 = []
    regret_gpts_C3 = []

    # GPUCB
    gpucb_learner_C1 = GPUCB_Learner(C=C[0], n_arms=n_arms_bids, arms=bids)
    reward_gpucb_C1 = []
    regret_gpucb_C1 = []

    gpucb_learner_C2 = GPUCB_Learner(C=C[1], n_arms=n_arms_bids, arms=bids)
    reward_gpucb_C2 = []
    regret_gpucb_C2 = []

    gpucb_learner_C3 = GPUCB_Learner(C=C[2], n_arms=n_arms_bids, arms=bids)
    reward_gpucb_C3 = []
    regret_gpucb_C3 = []

    for t in range(0, T):
        # Randomly select the context (ie.e., ['C1', 'C2', 'C3'] to which the user belongs) as it is known a priori
        context = np.random.choice(C)

        # GPTS
        if context == C[0]:
            pulled_arm = gpts_learner_C1.pull_arm()
            opt_price = opt_price_C1
        elif context == C[1]:
            pulled_arm = gpts_learner_C2.pull_arm()
            opt_price = opt_price_C2
        else:
            pulled_arm = gpts_learner_C3.pull_arm()
            opt_price = opt_price_C3

        reward = env.round(C=context, bid=bids[pulled_arm], price=opt_price)

        if context == C[0]:
            gpts_learner_C1.update(pulled_arm, reward)
            reward_gpts_C1.append(reward)
            reward_instant_gpts_C1[j].append(reward)

        elif context == C[1]:
            gpts_learner_C2.update(pulled_arm, reward)
            reward_gpts_C2.append(reward)
            reward_instant_gpts_C2[j].append(reward)

        else:
            gpts_learner_C3.update(pulled_arm, reward)
            reward_gpts_C3.append(reward)
            reward_instant_gpts_C3[j].append(reward)

        # GPUCB
        if context == C[0]:
            pulled_arm = gpucb_learner_C1.pull_arm()
            opt_price = opt_price_C1
        elif context == C[1]:
            pulled_arm = gpucb_learner_C2.pull_arm()
            opt_price = opt_price_C2
        else:
            pulled_arm = gpucb_learner_C3.pull_arm()
            opt_price = opt_price_C3

        reward = env.round(C=context, bid=bids[pulled_arm], price=opt_price)

        if context == C[0]:
            gpucb_learner_C1.update(pulled_arm, reward)
            reward_gpucb_C1.append(reward)
            reward_instant_gpucb_C1[j].append(reward)

        elif context == C[1]:
            gpucb_learner_C2.update(pulled_arm, reward)
            reward_gpucb_C2.append(reward)
            reward_instant_gpucb_C2[j].append(reward)

        else:
            gpucb_learner_C3.update(pulled_arm, reward)
            reward_gpucb_C3.append(reward)
            reward_instant_gpucb_C3[j].append(reward)

    # optimal reward from the experiment
    opt_C1 = max(max(reward_gpucb_C1), max(reward_gpts_C1))
    opt_array_C1 = np.full_like(reward_gpucb_C1, opt_C1)

    # optimal reward from the experiment
    opt_C2 = max(max(reward_gpucb_C2), max(reward_gpts_C2))
    opt_array_C2 = np.full_like(reward_gpucb_C2, opt_C2)

    # optimal reward from the experiment
    opt_C3 = max(max(reward_gpucb_C3), max(reward_gpts_C3))
    opt_array_C3 = np.full_like(reward_gpucb_C3, opt_C3)

    # GPTS
    reward_cum_gpts_C1[j] = np.cumsum(reward_gpts_C1)
    regret_cum_gpts_C1[j] = np.cumsum(opt_array_C1 - reward_gpts_C1)
    regret_instant_gpts_C1[j] = opt_array_C1 - reward_gpts_C1

    reward_cum_gpts_C2[j] = np.cumsum(reward_gpts_C2)
    regret_cum_gpts_C2[j] = np.cumsum(opt_array_C2 - reward_gpts_C2)
    regret_instant_gpts_C2[j] = opt_array_C2 - reward_gpts_C2

    reward_cum_gpts_C3[j] = np.cumsum(reward_gpts_C3)
    regret_cum_gpts_C3[j] = np.cumsum(opt_array_C3 - reward_gpts_C3)
    regret_instant_gpts_C3[j] = opt_array_C3 - reward_gpts_C3

    # GPUCB
    reward_cum_gpucb_C1[j] = np.cumsum(reward_gpucb_C1)
    regret_cum_gpucb_C1[j] = np.cumsum(opt_array_C1 - reward_gpucb_C1)
    regret_instant_gpucb_C1[j] = opt_array_C1 - reward_gpucb_C1

    reward_cum_gpucb_C2[j] = np.cumsum(reward_gpucb_C2)
    regret_cum_gpucb_C2[j] = np.cumsum(opt_array_C2 - reward_gpucb_C2)
    regret_instant_gpucb_C2[j] = opt_array_C2 - reward_gpucb_C2

    reward_cum_gpucb_C3[j] = np.cumsum(reward_gpucb_C3)
    regret_cum_gpucb_C3[j] = np.cumsum(opt_array_C3 - reward_gpucb_C3)
    regret_instant_gpucb_C3[j] = opt_array_C3 - reward_gpucb_C3

# Reshaping of the collected KPIs to plot them - FOR PRICING
''' C1 '''
min_length_C1 = 1000
for i in range(n_exp):
    if min_length_C1 > len(reward_instant_ts_C1[i]):
        min_length_C1 = len(reward_instant_ts_C1[i])

reward_instant_ts_C1_array = np.zeros((n_exp, min_length_C1))
regret_instant_ts_C1_array = np.zeros((n_exp, min_length_C1))
reward_cum_ts_C1_array = np.zeros((n_exp, min_length_C1))
regret_cum_ts_C1_array = np.zeros((n_exp, min_length_C1))

for i in range(n_exp):
    reward_instant_ts_C1_array[i] = reward_instant_ts_C1[i][:min_length_C1]
    regret_instant_ts_C1_array[i] = regret_instant_ts_C1[i][:min_length_C1]
    reward_cum_ts_C1_array[i] = reward_cum_ts_C1[i][:min_length_C1]
    regret_cum_ts_C1_array[i] = regret_cum_ts_C1[i][:min_length_C1]

''' C2 '''
min_length_C2 = 1000
for i in range(n_exp):
    if min_length_C2 > len(reward_instant_ts_C2[i]):
        min_length_C2 = len(reward_instant_ts_C2[i])

reward_instant_ts_C2_array = np.zeros((n_exp, min_length_C2))
regret_instant_ts_C2_array = np.zeros((n_exp, min_length_C2))
reward_cum_ts_C2_array = np.zeros((n_exp, min_length_C2))
regret_cum_ts_C2_array = np.zeros((n_exp, min_length_C2))

for i in range(n_exp):
    reward_instant_ts_C2_array[i] = reward_instant_ts_C2[i][:min_length_C2]
    regret_instant_ts_C2_array[i] = regret_instant_ts_C2[i][:min_length_C2]
    reward_cum_ts_C2_array[i] = reward_cum_ts_C2[i][:min_length_C2]
    regret_cum_ts_C2_array[i] = regret_cum_ts_C2[i][:min_length_C2]

''' C3 '''
min_length_C3 = 1000
for i in range(n_exp):
    if min_length_C3 > len(reward_instant_ts_C3[i]):
        min_length_C3 = len(reward_instant_ts_C3[i])

reward_instant_ts_C3_array = np.zeros((n_exp, min_length_C3))
regret_instant_ts_C3_array = np.zeros((n_exp, min_length_C3))
reward_cum_ts_C3_array = np.zeros((n_exp, min_length_C3))
regret_cum_ts_C3_array = np.zeros((n_exp, min_length_C3))

for i in range(n_exp):
    reward_instant_ts_C3_array[i] = reward_instant_ts_C3[i][:min_length_C3]
    regret_instant_ts_C3_array[i] = regret_instant_ts_C3[i][:min_length_C3]
    reward_cum_ts_C3_array[i] = reward_cum_ts_C3[i][:min_length_C3]
    regret_cum_ts_C3_array[i] = regret_cum_ts_C3[i][:min_length_C3]

# Reshaping of the collected KPIs to plot them - FOR ADVERTISING
''' C1 - GPTS '''
min_length_C1 = 1000
for i in range(n_exp):
    if min_length_C1 > len(reward_instant_gpts_C1[i]):
        min_length_C1 = len(reward_instant_gpts_C1[i])

reward_instant_gpts_C1_array = np.zeros((n_exp, min_length_C1))
regret_instant_gpts_C1_array = np.zeros((n_exp, min_length_C1))
reward_cum_gpts_C1_array = np.zeros((n_exp, min_length_C1))
regret_cum_gpts_C1_array = np.zeros((n_exp, min_length_C1))

for i in range(n_exp):
    reward_instant_gpts_C1_array[i] = reward_instant_gpts_C1[i][:min_length_C1]
    regret_instant_gpts_C1_array[i] = regret_instant_gpts_C1[i][:min_length_C1]
    reward_cum_gpts_C1_array[i] = reward_cum_gpts_C1[i][:min_length_C1]
    regret_cum_gpts_C1_array[i] = regret_cum_gpts_C1[i][:min_length_C1]

''' C1 - GPUCB '''
min_length_C1 = 1000
for i in range(n_exp):
    if min_length_C1 > len(reward_instant_gpucb_C1[i]):
        min_length_C1 = len(reward_instant_gpucb_C1[i])

reward_instant_gpucb_C1_array = np.zeros((n_exp, min_length_C1))
regret_instant_gpucb_C1_array = np.zeros((n_exp, min_length_C1))
reward_cum_gpucb_C1_array = np.zeros((n_exp, min_length_C1))
regret_cum_gpucb_C1_array = np.zeros((n_exp, min_length_C1))

for i in range(n_exp):
    reward_instant_gpucb_C1_array[i] = reward_instant_gpucb_C1[i][:min_length_C1]
    regret_instant_gpucb_C1_array[i] = regret_instant_gpucb_C1[i][:min_length_C1]
    reward_cum_gpucb_C1_array[i] = reward_cum_gpucb_C1[i][:min_length_C1]
    regret_cum_gpucb_C1_array[i] = regret_cum_gpucb_C1[i][:min_length_C1]

''' C2 - GPTS '''
min_length_C2 = 1000
for i in range(n_exp):
    if min_length_C2 > len(reward_instant_gpts_C2[i]):
        min_length_C2 = len(reward_instant_gpts_C2[i])

reward_instant_gpts_C2_array = np.zeros((n_exp, min_length_C2))
regret_instant_gpts_C2_array = np.zeros((n_exp, min_length_C2))
reward_cum_gpts_C2_array = np.zeros((n_exp, min_length_C2))
regret_cum_gpts_C2_array = np.zeros((n_exp, min_length_C2))

for i in range(n_exp):
    reward_instant_gpts_C2_array[i] = reward_instant_gpts_C2[i][:min_length_C2]
    regret_instant_gpts_C2_array[i] = regret_instant_gpts_C2[i][:min_length_C2]
    reward_cum_gpts_C2_array[i] = reward_cum_gpts_C2[i][:min_length_C2]
    regret_cum_gpts_C2_array[i] = regret_cum_gpts_C2[i][:min_length_C2]

''' C2 - GPUCB '''
min_length_C2 = 1000
for i in range(n_exp):
    if min_length_C2 > len(reward_instant_gpucb_C2[i]):
        min_length_C2 = len(reward_instant_gpucb_C2[i])

reward_instant_gpucb_C2_array = np.zeros((n_exp, min_length_C2))
regret_instant_gpucb_C2_array = np.zeros((n_exp, min_length_C2))
reward_cum_gpucb_C2_array = np.zeros((n_exp, min_length_C2))
regret_cum_gpucb_C2_array = np.zeros((n_exp, min_length_C2))

for i in range(n_exp):
    reward_instant_gpucb_C2_array[i] = reward_instant_gpucb_C2[i][:min_length_C2]
    regret_instant_gpucb_C2_array[i] = regret_instant_gpucb_C2[i][:min_length_C2]
    reward_cum_gpucb_C2_array[i] = reward_cum_gpucb_C2[i][:min_length_C2]
    regret_cum_gpucb_C2_array[i] = regret_cum_gpucb_C2[i][:min_length_C2]

''' C3 - GPTS '''
min_length_C3 = 1000
for i in range(n_exp):
    if min_length_C3 > len(reward_instant_gpts_C3[i]):
        min_length_C3 = len(reward_instant_gpts_C3[i])

reward_instant_gpts_C3_array = np.zeros((n_exp, min_length_C3))
regret_instant_gpts_C3_array = np.zeros((n_exp, min_length_C3))
reward_cum_gpts_C3_array = np.zeros((n_exp, min_length_C3))
regret_cum_gpts_C3_array = np.zeros((n_exp, min_length_C3))

for i in range(n_exp):
    reward_instant_gpts_C3_array[i] = reward_instant_gpts_C3[i][:min_length_C3]
    regret_instant_gpts_C3_array[i] = regret_instant_gpts_C3[i][:min_length_C3]
    reward_cum_gpts_C3_array[i] = reward_cum_gpts_C3[i][:min_length_C3]
    regret_cum_gpts_C3_array[i] = regret_cum_gpts_C3[i][:min_length_C3]

''' C3 - GPUCB '''
min_length_C3 = 1000
for i in range(n_exp):
    if min_length_C3 > len(reward_instant_gpucb_C3[i]):
        min_length_C3 = len(reward_instant_gpucb_C3[i])

reward_instant_gpucb_C3_array = np.zeros((n_exp, min_length_C3))
regret_instant_gpucb_C3_array = np.zeros((n_exp, min_length_C3))
reward_cum_gpucb_C3_array = np.zeros((n_exp, min_length_C3))
regret_cum_gpucb_C3_array = np.zeros((n_exp, min_length_C3))

for i in range(n_exp):
    reward_instant_gpucb_C3_array[i] = reward_instant_gpucb_C3[i][:min_length_C3]
    regret_instant_gpucb_C3_array[i] = regret_instant_gpucb_C3[i][:min_length_C3]
    reward_cum_gpucb_C3_array[i] = reward_cum_gpucb_C3[i][:min_length_C3]
    regret_cum_gpucb_C3_array[i] = regret_cum_gpucb_C3[i][:min_length_C3]

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values for price learning in contexts C1, C2, C3')
axs[0, 0].plot(np.mean(regret_cum_ts_C1_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ts_C2_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_ts_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_ts_C1_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ts_C2_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_ts_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_ts_C1_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ts_C2_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_ts_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_ts_C1_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ts_C2_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_ts_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Standard deviations for price learning in contexts C1, C2, C3')
axs[0, 0].plot(np.std(regret_cum_ts_C1_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_ts_C2_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_ts_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_ts_C1_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_ts_C2_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_ts_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_ts_C1_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_ts_C2_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_ts_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_ts_C1_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_ts_C2_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_ts_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values for advertising (GPTS) in contexts C1, C2, C3')
axs[0, 0].plot(np.mean(regret_cum_gpts_C1_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_gpts_C2_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_gpts_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_gpts_C1_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_gpts_C2_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_gpts_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_gpts_C1_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_gpts_C2_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_gpts_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_gpts_C1_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_gpts_C2_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_gpts_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle(
    'Standard deviations for advertising (GPTS) in contexts C1, C2, C3')
axs[0, 0].plot(np.std(regret_cum_gpts_C1_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_gpts_C2_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_gpts_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_gpts_C1_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_gpts_C2_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_gpts_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_gpts_C1_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_gpts_C2_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_gpts_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_gpts_C1_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_gpts_C2_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_gpts_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the average value of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle('Average values for advertising (GPUCB) in contexts C1, C2, C3')
axs[0, 0].plot(np.mean(regret_cum_gpucb_C1_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_gpucb_C2_array, axis=0))
axs[0, 0].plot(np.mean(regret_cum_gpucb_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.mean(reward_cum_gpucb_C1_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_gpucb_C2_array, axis=0))
axs[0, 1].plot(np.mean(reward_cum_gpucb_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.mean(regret_instant_gpucb_C1_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_gpucb_C2_array, axis=0))
axs[1, 0].plot(np.mean(regret_instant_gpucb_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.mean(reward_instant_gpucb_C1_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_gpucb_C2_array, axis=0))
axs[1, 1].plot(np.mean(reward_instant_gpucb_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')

# Plots of the standard deviation of the cumulative regret, cumulative reward, instantaneous regret and instantaneous reward
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle(
    'Standard deviations for advertising (GPUCB) in contexts C1, C2, C3')
axs[0, 0].plot(np.std(regret_cum_gpucb_C1_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_gpucb_C2_array, axis=0))
axs[0, 0].plot(np.std(regret_cum_gpucb_C3_array, axis=0))
axs[0, 0].legend(['C1', 'C2', 'C3'])
axs[0, 0].set_title('Cumulative regret')
axs[0, 1].plot(np.std(reward_cum_gpucb_C1_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_gpucb_C2_array, axis=0))
axs[0, 1].plot(np.std(reward_cum_gpucb_C3_array, axis=0))
axs[0, 1].legend(['C1', 'C2', 'C3'])
axs[0, 1].set_title('Cumulative reward')
axs[1, 0].plot(np.std(regret_instant_gpucb_C1_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_gpucb_C2_array, axis=0))
axs[1, 0].plot(np.std(regret_instant_gpucb_C3_array, axis=0))
axs[1, 0].legend(['C1', 'C2', 'C3'])
axs[1, 0].set_title('Instantaneous regret')
axs[1, 1].plot(np.std(reward_instant_gpucb_C1_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_gpucb_C2_array, axis=0))
axs[1, 1].plot(np.std(reward_instant_gpucb_C3_array, axis=0))
axs[1, 1].legend(['C1', 'C2', 'C3'])
axs[1, 1].set_title('Instantaneous reward')
