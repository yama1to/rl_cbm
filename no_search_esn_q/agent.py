"""
2021/09/24
強化学習のエージェント
"""

import numpy as np
import matplotlib.pyplot as plt


class Agent():

    def __init__(self):
        self.output_train = {}
        self.output_test = {}
        self.reward_log_train = []
        self.reward_log_test = []

    #報酬
    def init_log_train(self):
        self.reward_log_train = []
    
    def init_log_test(self):
        self.reward_log_test = []

    #獲得した報酬の記録
    def log_train(self, reward):
        self.reward_log_train.append(reward)

    def log_test(self, reward):
        self.reward_log_test.append(reward)

    #報酬の表示
    def show_reward_log_train(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log_train[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log_train), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log_train[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward Train History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()

    
    def show_reward_log_test(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log_test[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log_test), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log_test[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward Test History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
