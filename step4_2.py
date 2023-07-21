from matplotlib import contextlib
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
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
from step1 import *
from step2 import *
from step3 import*
from step4_1 import*

# Define the GP-UCB learner


class GPUCB_Learner:
    def __init__(self, n_arms, arms):
        self.n_arms = n_arms
        self.arms = arms
        self.times_pulled = np.zeros(n_arms)
        self.collected_rewards = np.zeros(n_arms)
        self.ucb_values = np.zeros(n_arms)
        self.regret = []
        self.context_estimation = {}  # To store the estimated context

    def pull_arm(self, context):
        # Calculate the UCB values for each arm based on the estimated context
        self.ucb_values = self.collected_rewards / self.times_pulled + np.sqrt(
            (2 * np.log(context['t'])) / self.times_pulled)
        return np.argmax(self.ucb_values)

    def update(self, pulled_arm, reward, context):
        self.times_pulled[pulled_arm] += 1
        self.collected_rewards[pulled_arm] += reward
        # Update the estimated context based on features and rewards

        self.regret.append(max(self.collected_rewards.flatten()) - reward)


# Define the GP-TS learner
class GPTS_Learner:
    def __init__(self, n_arms, arms):
        self.n_arms = n_arms
        self.arms = arms
        self.beta_parameters = np.ones((n_arms, 2))
        self.samples = np.zeros(n_arms_bids)
        self.regret = []
        self.context_estimation = {}  # To store the estimated context

    def pull_arm(self, context):
        clipped_beta_parameters = np.clip(self.beta_parameters, 1e-6, None)
        sampled_rates = np.random.beta(
            clipped_beta_parameters[:, 0], clipped_beta_parameters[:, 1])
        return np.argmax(sampled_rates)

    def update(self, pulled_arm, reward, context):
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1 - reward
        # Update the estimated context based on features and rewards

        self.regret.append(np.max(self.samples) - reward)

# Definition of the new environment:
# Define the Environment


class Environment_Dynamic:
    def __init__(self, unit_cost, n_arms_prices, n_arms_bids, prob_prices, prob_bids, bids, sigma):
        self.n_arms_prices = n_arms_prices
        self.n_arms_bids = n_arms_bids
        self.bids = bids
        self.sigmas = np.ones(len(bids)) * sigma
        self.unit_cost = unit_cost
        self.user_classes = []
        self.current_time_step = 0
        # Set user classes
        self.set_user_classes(prob_prices, prob_bids)

    def set_user_classes(self, prob_prices, prob_bids):
        self.user_classes = [
            UserClass(clicks_func=lambda bid: 10 * np.exp(-((bid - 0.5) / 1.5) ** 2),
                      cost_func=lambda bid: bid * 10 *
                      (np.exp(-((bid - 0.5) / 1.5) ** 2)),
                      conversion_prob_func=lambda price: 0.8 *
                      (1 + np.exp(-price)),
                      f1=0, f2=0, prob_prices=prob_prices, prob_bids=prob_bids),
            UserClass(clicks_func=lambda bid: 15 * np.exp(-((bid - 0.3) / 1.5) ** 2),
                      cost_func=lambda bid: bid * 15 *
                      (np.exp(-((bid - 0.3) / 1.5) ** 2)),
                      conversion_prob_func=lambda price: 0.7 *
                      (1 + np.exp(-price)),
                      f1=0, f2=1, prob_prices=prob_prices, prob_bids=prob_bids),
            UserClass(clicks_func=lambda bid: 12 * np.exp(-((bid - 0.7) / 1.5) ** 2),
                      cost_func=lambda bid: bid * 12 *
                      (np.exp(-((bid - 0.7) / 1.5) ** 2)),
                      conversion_prob_func=lambda price: 0.6 *
                      (1 + np.exp(-price)),
                      f1=1, f2=0, prob_prices=prob_prices, prob_bids=prob_bids)
        ]

    def generate_context(self):
        features = self.get_features()

        user_class = self.get_user_class(features)
        bid = features['bid']
        price = features['price']
        t = self.current_time_step  # Add the current time step 't' to the context
        optimal_reward = self.round(user_class, bid, price)

        context = {
            'F1': user_class.f1,
            'F2': user_class.f2,
            'bid': bid,
            'price': price,
            't': t,  # Add the current time step 't' to the context
            'optimal_reward': optimal_reward
        }

        return context

    def get_user_class(self, features):
        for user_class in self.user_classes:
            if user_class.check_features(features):
                return user_class
        return None

    def daily_clicks(self, user_class, bid):

        clicks_func = user_class.clicks_func
        clicks = clicks_func(bid) + np.random.normal(0, 0.2)
        return max(0, clicks)

    def daily_clicks_cost(self, user_class, bid):
        cost_func = user_class.cost_func
        cost = cost_func(bid) + np.random.normal(0, 0.1)
        return max(0, cost)

    def conversion_prob(self, user_class, price):
        conversion_prob_func = user_class.conversion_prob_func
        conversion_prob = conversion_prob_func(
            price) + np.random.normal(0, 0.05)
        return min(1, max(0, conversion_prob))

    def round(self, user_class, bid, price):
        reward = self.daily_clicks(user_class, bid) * self.conversion_prob(user_class, price) * (
            price - self.unit_cost) - self.daily_clicks_cost(user_class, bid)
        return reward

    def get_features(self):
        features = {}
        features['F1'] = random.choice([0, 1])
        if features['F1'] == 1:
            features['F2'] = 0
        else:
            features['F2'] = random.choice([0, 1])

        # Randomly select a bid from the available bids
        features['bid'] = np.random.choice(self.bids)
        # Randomly select a price from the available prices
        features['price'] = np.random.choice(self.n_arms_prices)

        return features


# Define the UserClass
class UserClass:
    def __init__(self, clicks_func, cost_func, conversion_prob_func, f1, f2, prob_prices, prob_bids):
        self.clicks_func = clicks_func
        self.cost_func = cost_func
        self.conversion_prob_func = conversion_prob_func
        self.f1 = f1
        self.f2 = f2
        self.prob_prices = prob_prices
        self.prob_bids = prob_bids

    def check_features(self, features):

        return features['F1'] == self.f1 and features['F2'] == self.f2


# Define the function to run the experiments with context generation every two weeks


def preprocess_context(context_data):
    le = LabelEncoder()
    preprocessed_context = {}
    for key in context_data[0].keys():
        preprocessed_context[key] = le.fit_transform(
            [data[key] for data in context_data])
    return np.column_stack(list(preprocessed_context.values()))


def plot_data_points(context_data):
    f1_values = [data['F1'] for data in context_data]
    f2_values = [data['F2'] for data in context_data]
    bids = [data['bid'] for data in context_data]
    prices = [data['price'] for data in context_data]

    # Create a 2D plot
    plt.figure(figsize=(10, 6))
    plt.scatter(f1_values, f2_values, c='blue', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Plot of Context Data')
    plt.legend()
    plt.show()

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(f1_values, f2_values, bids, c='blue', label='Data Points')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Bid')
    ax.set_title('3D Plot of Context Data')
    plt.legend()
    plt.show()


def run_experiment_with_context_generation(env, num_runs=10, horizon=1000):
    avg_cumulative_regret_gpucb = np.zeros((num_runs, horizon + 1))
    avg_cumulative_reward_gpucb = np.zeros((num_runs, horizon + 1))
    avg_cumulative_regret_gpts = np.zeros((num_runs, horizon + 1))
    avg_cumulative_reward_gpts = np.zeros((num_runs, horizon + 1))

    avg_insta_regret_gpucb = []
    avg_insta_reward_gpucb = []
    avg_insta_regret_gpts = []
    avg_insta_reward_gpts = []
    context_data = []

    for _ in range(num_runs):
        gpucb_learner_start = GPUCB_Learner(
            n_arms=env.n_arms_bids, arms=env.bids)
        gpts_learner_start = GPTS_Learner(
            n_arms=env.n_arms_bids, arms=env.bids)

        cumulative_regret_gpucb = [0]
        cumulative_reward_gpucb = [0]
        cumulative_regret_gpts = [0]
        cumulative_reward_gpts = [0]
        instantaneous_regret_gpucb = []
        instantaneous_reward_gpucb = []
        instantaneous_regret_gpts = []
        instantaneous_reward_gpts = []

        for t in range(1, 3):
            env.current_time_step = t
            context = env.generate_context()
            features = env.get_features()
            context_data.append(context)
            user_class = env.get_user_class(context)
            bid = features['bid']
            price = features['price']
            pulled_arm_gpucb = gpucb_learner_start.pull_arm(context)
            pulled_arm_gpts = gpts_learner_start.pull_arm(context)
            reward_gpucb = env.round(
                user_class, env.bids[pulled_arm_gpucb], price)
            reward_gpts = env.round(
                user_class, env.bids[pulled_arm_gpts], price)
            gpucb_learner_start.update(pulled_arm_gpucb, reward_gpucb, context)
            gpts_learner_start.update(pulled_arm_gpts, reward_gpts, context)
            optimal_reward = env.round(user_class, bid, price)
            cumulative_regret_gpucb.append(
                cumulative_regret_gpucb[-1] + optimal_reward - reward_gpucb)
            cumulative_reward_gpucb.append(
                cumulative_reward_gpucb[-1] + reward_gpucb)
            cumulative_regret_gpts.append(
                cumulative_regret_gpts[-1] + optimal_reward - reward_gpts)
            cumulative_reward_gpts.append(
                cumulative_reward_gpts[-1] + reward_gpts)
            instantaneous_regret_gpucb.append(optimal_reward - reward_gpucb)
            instantaneous_reward_gpucb.append(reward_gpucb)
            instantaneous_regret_gpts.append(optimal_reward - reward_gpts)
            instantaneous_reward_gpts.append(reward_gpts)
        print(context_data[-3:])
        context_array = preprocess_context(context_data[-3:])
        plot_data_points(context_data)
        print('This is the context data')
        print(context_data)
        print('This is it transformed')
        print(context_array)
        clustering = DBSCAN(eps=0.5, min_samples=100).fit(context_array)
        n_clusters_ = len(set(clustering.labels_)) - \
            (1 if -1 in clustering.labels_ else 0)
        print('Number of clusters')
        print(n_clusters_)
        learner_dictionaryGPUCB = {}
        learner_dictionaryGPTS = {}

        for cluster_label in np.unique(clustering.labels_):

            if cluster_label == -1:
                continue
            learner_dictionaryGPUCB[cluster_label] = GPUCB_Learner(
                n_arms=env.n_arms_bids, arms=env.bids)
            learner_dictionaryGPTS[cluster_label] = GPTS_Learner(
                n_arms=env.n_arms_bids, arms=env.bids)

        cumulative_regret_gpucb = [0]
        cumulative_reward_gpucb = [0]
        cumulative_regret_gpts = [0]
        cumulative_reward_gpts = [0]

        for t in range(3, horizon + 1):
            if t % 14 == 0:
                env.current_time_step = t
                context = env.generate_context()
                features = env.get_features()
                context_data.append(context)

                context_array = preprocess_context(context_data[-14:])

                clustering = DBSCAN(
                    eps=0.5, min_samples=100).fit(context_array)
                _clusters_ = len(set(clustering.labels_)) - \
                    (1 if -1 in clustering.labels_ else 0)
                print('Number of clusters')
                print(n_clusters_)
                learner_dictionaryGPUCB = {}
                learner_dictionaryGPTS = {}
                for cluster_label in np.unique(clustering.labels_):
                    if cluster_label == -1:
                        continue
                    learner_dictionaryGPUCB[cluster_label] = GPUCB_Learner(
                        n_arms=env.n_arms_bids, arms=env.bids)
                    learner_dictionaryGPTS[cluster_label] = GPTS_Learner(
                        n_arms=env.n_arms_bids, arms=env.bids)

        for cluster_label in np.unique(clustering.labels_):
            if cluster_label == -1:
                continue

            user_class = env.get_user_class(context)
            bid = features['bid']
            price = features['price']

            pulled_arm_gpucb = learner_dictionaryGPUCB[cluster_label].pull_arm(
                context)
            pulled_arm_gpts = learner_dictionaryGPTS[cluster_label].pull_arm(
                context)
            reward_gpucb = env.round(
                user_class, env.bids[pulled_arm_gpucb], price)
            reward_gpts = env.round(
                user_class, env.bids[pulled_arm_gpts], price)
            learner_dictionaryGPUCB[cluster_label].update(
                pulled_arm_gpucb, reward_gpucb, context)
            learner_dictionaryGPTS[cluster_label].update(
                pulled_arm_gpts, reward_gpts, context)
            optimal_reward = env.round(user_class, bid, price)
            cumulative_regret_gpucb.append(
                cumulative_regret_gpucb[-1] + optimal_reward - reward_gpucb)
            cumulative_reward_gpucb.append(
                cumulative_reward_gpucb[-1] + reward_gpucb)
            cumulative_regret_gpts.append(
                cumulative_regret_gpts[-1] + optimal_reward - reward_gpts)
            cumulative_reward_gpts.append(
                cumulative_reward_gpts[-1] + reward_gpts)
            instantaneous_regret_gpucb.append(optimal_reward - reward_gpucb)
            instantaneous_reward_gpucb.append(reward_gpucb)
            instantaneous_regret_gpts.append(optimal_reward - reward_gpts)
            instantaneous_reward_gpts.append(reward_gpts)

        avg_cumulative_regret_gpucb[_] = cumulative_regret_gpucb
        avg_cumulative_reward_gpucb[_] = cumulative_reward_gpucb
        avg_cumulative_regret_gpts[_] = cumulative_regret_gpts
        avg_cumulative_reward_gpts[_] = cumulative_reward_gpts

        avg_insta_regret_gpucb.append(instantaneous_regret_gpucb)
        avg_insta_reward_gpucb.append(instantaneous_reward_gpucb)
        avg_insta_regret_gpts.append(instantaneous_regret_gpts)
        avg_insta_reward_gpts.append(instantaneous_reward_gpts)

    avg_cumulative_regret_gpucb /= num_runs
    avg_cumulative_reward_gpucb /= num_runs
    avg_cumulative_regret_gpts /= num_runs
    avg_cumulative_reward_gpts /= num_runs

    mean_insta_regret_gpucb = np.mean(avg_insta_regret_gpucb, axis=0)
    std_insta_regret_gpucb = np.std(avg_insta_regret_gpucb, axis=0)
    mean_insta_reward_gpucb = np.mean(avg_insta_reward_gpucb, axis=0)
    std_insta_reward_gpucb = np.std(avg_insta_reward_gpucb, axis=0)

    mean_insta_regret_gpts = np.mean(avg_insta_regret_gpts, axis=0)
    std_insta_regret_gpts = np.std(avg_insta_regret_gpts, axis=0)
    mean_insta_reward_gpts = np.mean(avg_insta_reward_gpts, axis=0)
    std_insta_reward_gpts = np.std(avg_insta_reward_gpts, axis=0)

    return avg_cumulative_regret_gpucb, avg_cumulative_reward_gpucb, avg_cumulative_regret_gpts, avg_cumulative_reward_gpts, \
        mean_insta_regret_gpucb, std_insta_regret_gpucb, mean_insta_reward_gpucb, std_insta_reward_gpucb, \
        mean_insta_regret_gpts, std_insta_regret_gpts, mean_insta_reward_gpts, std_insta_reward_gpts


# Define the parameters

unit_cost = 1
n_arms_bids = 100
bids = np.linspace(0.0, 1.0, n_arms_bids)
prob_bids = np.random.random_sample(n_arms_bids)
n_arms_prices = 5
prob_prices = np.random.random_sample(n_arms_prices)
sigma = 0.05
n_exp = 10
T = 365

# Initialize the environment
env = Environment_Dynamic(unit_cost=unit_cost, n_arms_prices=n_arms_prices, n_arms_bids=n_arms_bids,
                          prob_prices=prob_prices, prob_bids=prob_bids, bids=bids, sigma=sigma)

# Run experiments with context generation every two weeks
avg_cumulative_regret_gpucb, avg_cumulative_reward_gpucb, avg_cumulative_regret_gpts, avg_cumulative_reward_gpts, \
    mean_insta_regret_gpucb, std_insta_regret_gpucb, mean_insta_reward_gpucb, std_insta_reward_gpucb, \
    mean_insta_regret_gpts, std_insta_regret_gpts, mean_insta_reward_gpts, std_insta_reward_gpts = run_experiment_with_context_generation(
        env, num_runs=n_exp, horizon=T)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(abs(avg_cumulative_regret_gpucb), label='GP-UCB')
plt.plot(abs(avg_cumulative_regret_gpts), label='GP-TS')
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.title('Average Cumulative Regret with Context Generation')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(avg_cumulative_reward_gpucb, label='GP-UCB')
plt.plot(avg_cumulative_reward_gpts, label='GP-TS')
plt.xlabel('Time')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.title('Average Cumulative Reward with Context Generation')
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(range(1, 365 + 1), mean_insta_regret_gpucb /
         std_insta_regret_gpucb, label='GP-UCB')
plt.plot(range(1, 365 + 1), mean_insta_regret_gpts /
         std_insta_regret_gpts, label='GP-TS')
plt.xlabel('Time')
plt.ylabel('Standardized Instantaneous Regret')
plt.legend()
plt.title('Standardized Average Instantaneous Regret')
plt.show()
# Plot standardized average instantaneous reward
plt.figure(figsize=(10, 6))
plt.plot(range(1, 365 + 1),  mean_insta_reward_gpucb /
         std_insta_reward_gpucb, label='GP-UCB')
plt.plot(range(1, 365 + 1), mean_insta_reward_gpts /
         std_insta_reward_gpts, label='GP-TS')
plt.xlabel('Time')
plt.ylabel('Standardized Instantaneous Reward')
plt.legend()
plt.title('Standardized Average Instantaneous Reward')
plt.show()
