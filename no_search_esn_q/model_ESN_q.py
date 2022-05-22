"""
2021/12/15
frozenlakeを解くために実装したESN
Q-learningを実装
ε-greedy法を実装
"""

import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_matrix_esn import *

class Hyperpm_ESN():
    def __init__(self):
        self.Nu = 16   #size of input
        self.Nx = 20#500 #size of dynamical reservior
        self.Ny = 4   #size of output
        
        # alphaは結合強度
        self.alpha_i = 0.6
        self.alpha_r = 1
        self.alpha_b = 0.

        # betaは結合荷重のスパース度
        self.beta_i = 0.8
        self.beta_r = 0.1
        self.beta_b = 0.0

        # 出力層の結合荷重の更新に関する値
        #-----------------------------------------------------------------

        

        self.eta_choice = 3
        self.eta_max_goal = 250
        self.gamma_wout = 0.9 # defaultは0.985最も良い0.925出力層の結合荷重の割引率γ
        self.k_greedy = 0.005 # ε-greedy法のεの決定係数
        #-----------------------------------------------------------------


class ESN_q(Hyperpm_ESN):
    
    def __init__(self):
        super().__init__()
        self.ep_ini = 0.1 # 出力層の結合荷重の更新の学習率η
        self.ep_fin = 0.0 # 出力層の結合荷重の更新の学習率η
        self.ep_2 = 0.01

        self.eta_ini = 0.1 # 出力層の結合荷重の更新の学習率η
        self.eta_fin = 0.0 # 出力層の結合荷重の更新の学習率η
        self.eta_2 = 0.01

        # 結合荷重の生成
        self.Wr = generate_random_matrix(self.Nx,self.Nx,self.alpha_r,self.beta_r,distribution="one",normalization="sr")
        self.Wb = generate_random_matrix(self.Nx,self.Ny,self.alpha_b,self.beta_b,distribution="one",normalization="none")
        self.Wi = generate_random_matrix(self.Nx,self.Nu,self.alpha_i,self.beta_i,distribution="one",normalization="none")
        self.Wo = np.zeros(self.Nx * self.Ny).reshape(self.Ny, self.Nx)

        self.X_collect = [] 
        self.R_collect = [] 
        self.U_collect = [] 
        self.Q_collect = []

        self.Epsilon_collect = [] #εを格納
        self.Eta_collect = []
        
        self.pre_episode_plcy = -1
        self.pre_episode_eta = -1
        self.begin_cnt_eta = 0
        self.all_stp_num = 0
        self.one_epsd_step_num = 0
        self.episode_num = 0
        self.goal_num = 0
        self.all_epsd = 0
        self.all_actions = 0
        self.learning_mode = 0


    def initialize_setting(self, actions, epsd_num, learning_mode):
        self.all_actions = actions
        self.all_epsd = epsd_num
        self.learning_mode = learning_mode


    def eta(self,rewards):
        recent_reward = np.sum(np.heaviside(np.array(rewards[-20:]),0))/20
        
        eta = self.eta_fin + (self.eta_ini - self.eta_fin) * np.exp(-recent_reward/self.eta_2)

        if self.pre_episode_eta != self.episode_num:
            self.Eta_collect.append(eta)
        
        self.pre_episode_eta = self.episode_num

        return eta
 

    # ε-greedy法を用いた戦略
    def policy_epsilon(self,rewards):
        recent_reward = np.sum(np.heaviside(np.array(rewards[-20:]),0))/20

        ep = self.ep_fin + (self.ep_ini - self.ep_fin) * np.exp(-recent_reward/self.ep_2)
       

        if np.random.random() < ep:
            next_action =  np.random.randint(len(self.all_actions))
        else:
            if sum(self.q) != 0:
                next_action = np.argmax(self.q)
            else:
                next_action =  np.random.randint(len(self.all_actions))
        
        return next_action


    # 1episode終わるごとにネットワークをリセットする
    def reset_network(self, current_episode, current_goal_num):
        self.one_epsd_step_num = 0
        self.episode_num = current_episode
        self.goal_num = current_goal_num
        #neuron
        self.x = np.zeros(self.Nx)
        self.x_next = np.zeros(self.Nx)
        #self.r = np.tanh(self.x)
        self.r = np.zeros(self.Nx)
        self.r_next = np.zeros(self.Nx)
        #output
        self.q = np.zeros(self.Ny)
        self.q_next = np.zeros(self.Ny)


    # Q値の計算
    def step_network_q(self, state):
        
        # 入力を状態のindexが1のOne-hotベクトルに変換
        change_input = np.zeros(self.Nu) #:TODO
        change_input[state] = 1    
        esn_input = np.tanh(change_input)

        #dt = 1.0
        sum_esn = np.zeros(self.Nx)
        sum_esn += self.Wi @ esn_input
        sum_esn += self.Wr @ self.r
        #sum_esn += self.Wb @ self.q

        #x_next = (1 -dt /self.tau_x)*self.x +dt/self.tau_x*(sum)
        self.x_next = sum_esn
        #r_next = np.tanh(self.beta*x_next)
        self.r_next = np.tanh(self.x_next)
        self.q_next = np.tanh(self.Wo @ self.r_next)

        ### Record
        if self.learning_mode == 1 and self.episode_num >= self.all_epsd-15:
            self.X_collect.append(self.x_next)
            self.R_collect.append(self.r_next)
            self.Q_collect.append(self.q_next)
            self.U_collect.append(esn_input)

        self.one_epsd_step_num += 1 #:TODO
        self.all_stp_num += 1 #:TODO


    ### training output conection
    def train_weight_q(self, action, reward,rewards):
        Wo_next = self.Wo
        Wo_next[action] = self.Wo[action]+self.eta(rewards)*\
            np.tanh(reward+self.gamma_wout*max(self.q_next)-self.q[action])*self.r

        ### Update
        self.Wo = Wo_next
        self.x = self.x_next
        self.r = self.r_next
        self.q = self.q_next


    ### Update
    def esn_update_q(self):
        self.x = self.x_next
        self.r = self.r_next
        self.q = self.q_next
        

    def plot_esn(self):
        fig = plt.figure(figsize=(20, 12))
        Nr = 3
        ax = fig.add_subplot(Nr,1,1)
        ax.cla()
        #ax.set_title("U")
        ax.plot(self.U_collect)

        # ax = fig.add_subplot(Nr,1,2)
        # ax.cla()
        # ax.set_title("X")
        # ax.plot(self.X_collect)

        ax = fig.add_subplot(Nr,1,2)
        ax.cla()
        #ax.set_title("X")
        ax.plot(self.R_collect)

        ax = fig.add_subplot(Nr,1,3)
        ax.cla()
        #ax.set_title("Q")
        ax.plot(self.Q_collect)

        plt.show()

    def plot_esn_out(self):
        fig = plt.figure(figsize=(20, 12))
        Nr = 3
        ax = fig.add_subplot(Nr,1,1)
        ax.cla()
        ax.set_title("Epsiron")
        ax.plot(self.Epsilon_collect)

        ax = fig.add_subplot(Nr,1,2)
        ax.cla()
        ax.set_title("Eta")
        ax.plot(self.Eta_collect)

        ax = fig.add_subplot(Nr,1,3)
        ax.cla()
        ax.set_title("Q")
        ax.plot(self.Q_collect)

        plt.show()