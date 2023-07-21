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

T = 365


class UserClass ():
    def __init__(self, clicks_func, cost_func, conversion_prob_func, f1, f2):
        # function that expresses the number of daily clicks as the bid varies
        self.clicks_func = clicks_func
        # function that assigns the cumulative daily cost of the clicks as the bid varies
        self.cost_func = cost_func
        # function expressing how the conversion probability varies as the price varies
        self.conversion_prob_func = conversion_prob_func
        self.f1 = f1
        self.f2 = f2


class Environment ():
    def __init__(self, unit_cost, n_arms_prices, n_arms_bids, bids, sigma):
        self.n_arms_prices = n_arms_prices
        self.n_arms_bids = n_arms_bids
        self.bids = bids
        self.sigmas = np.ones(len(bids))*sigma
        self.unit_cost = unit_cost  # Unitary cost (fixed)
        self.user_classes = {
            'C1': UserClass(clicks_func=lambda bid: 10*np.exp(-((bid - 0.5) / 1.5) ** 2),  # np.exp(-((x - center) / width) ** 2)
                            # Strictly positive concave curve which reaches a maximum at the center parameter and decreases symmetrically on both sides
                            # The width parameter controls the width of the curve
                            # The bid is approximately the price paid per click
                            cost_func=lambda bid: bid*10 * \
                            (np.exp(-((bid - 0.5) / 1.5) ** 2)),
                            # Prob that decreases with the price (min price: 2)
                            conversion_prob_func=lambda price: 0.8 * \
                            (1 + np.exp(-price)),
                            f1=0,
                            f2=0),

            'C2': UserClass(clicks_func=lambda bid: 15*np.exp(-((bid - 0.3) / 1.5) ** 2),
                            cost_func=lambda bid: bid*15 * \
                            (np.exp(-((bid - 0.3) / 1.5) ** 2)),
                            conversion_prob_func=lambda price: 0.7 * \
                            (1 + np.exp(-price)),
                            f1=0,
                            f2=1),

            'C3': UserClass(clicks_func=lambda bid: 12*np.exp(-((bid - 0.7) / 1.5) ** 2),
                            cost_func=lambda bid: bid*12 * \
                            (np.exp(-((bid - 0.7) / 1.5) ** 2)),
                            conversion_prob_func=lambda price: 0.6 * \
                            (1 + np.exp(-price)),
                            f1=1,
                            f2=0)}

    def daily_clicks(self, user_class, bid):
        clicks_func = self.user_classes[user_class].clicks_func
        clicks = clicks_func(bid) + np.random.normal(0, 0.2)
        return max(0, clicks)

    def daily_clicks_cost(self, user_class, bid):
        cost_func = self.user_classes[user_class].cost_func
        cost = cost_func(bid) + np.random.normal(0, 0.1)
        return max(0, cost)

    def conversion_prob(self, user_class, price):
        conversion_prob_func = self.user_classes[user_class].conversion_prob_func
        conversion_prob = conversion_prob_func(
            price) + np.random.normal(0, 0.05)
        return min(1, max(0, conversion_prob))

    def round(self, C, bid, price):  # Models the interaction with the learner
        reward = self.daily_clicks(C, bid) * self.conversion_prob(C, price) * (
            price - self.unit_cost) - self.daily_clicks_cost(C, bid)
        return reward


class Learner ():
    def __init__(self, C, n_arms, arms):
        self.user_class = C
        self.n_arms = n_arms
        self.arms = arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
