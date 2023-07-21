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

# Step 2: Learning for advertising

from environment import *
from step1 import *


class GPTS_Learner(Learner):
    def __init__(self, C, n_arms, arms):
        super().__init__(C, n_arms, arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        alpha = 5e-5
        kernel = CK(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=2)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        if len(y) > 1:
            self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)


# GPUCB_Learner is defined as a child class of the GPTS_Learner, the pull_arm function is overwritten
class GPUCB_Learner(GPTS_Learner):
    def __init__(self, C, n_arms, arms):
        super().__init__(C, n_arms, arms)

    def compute_ucb(self):
        delta = 0.1
        beta = math.sqrt(
            (2 * math.log((self.t ** 2) * math.pi ** 2 / (6 * delta))) / self.t)
        return self.means + beta * self.sigmas

    def pull_arm(self):
        # Play once every arm
        if self.t < self.n_arms:
            return self.t
        # Play the most promising arm
        else:
            ucb_values = self.compute_ucb()
            return np.argmax(ucb_values)


C = 'C1'
unit_cost = 1
n_arms = 100  # The seller can choose among 100 possible bids
bids = np.linspace(0.0, 1.0, n_arms)
sigma = 0.05
price = 6

n_exp = 3

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
                      n_arms_bids=n_arms, bids=bids, sigma=sigma)
    # GPTS
    gpts_learner = GPTS_Learner(C=C, n_arms=n_arms, arms=bids)
    reward_gpts = []
    regret_gpts = []
    # GPUCB
    gpucb_learner = GPUCB_Learner(C=C, n_arms=n_arms, arms=bids)
    reward_gpucb = []
    regret_gpucb = []

    for t in range(0, T):
        # GPTS
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(C=C, bid=bids[pulled_arm], price=price)
        gpts_learner.update(pulled_arm, reward)
        reward_gpts.append(reward)

        reward_instant_gpts[j, t] = reward

        # GPUCB
        pulled_arm = gpucb_learner.pull_arm()
        reward = env.round(C=C, bid=bids[pulled_arm], price=price)
        gpucb_learner.update(pulled_arm, reward)
        reward_gpucb.append(reward)

        reward_instant_gpucb[j, t] = reward

    # Optimal reward is set as the maximum reward from the experiment
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
