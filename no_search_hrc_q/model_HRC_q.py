"""
2021/12/15
frozenlakeを解くために実装したHRC
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
from generate_matrix_hrc import *

class Hyperpm_HRC():
    def __init__(self,c):

        self.NN = c.NN # 1サイクル当たりの時間ステップ

        self.Nu = c.Nu   # size of input
        self.Nh = c.Nh # size of dynamical reservior
        self.Ny = c.Ny   # size of output
        
        self.Temp = c.Temp
        self.dt = c.dt # 0.001

        #sigma_np = -5
        self.alpha_i = c.alpha_i
        self.alpha_r = c.alpha_r

        self.alpha_b = 0.
        self.alpha_s = c.alpha_s

        self.beta_i = c.beta_i
        self.beta_r = c.beta_r
        self.beta_b = 0.

        # 出力層の結合荷重の更新に関する値
        #-----------------------------------------------------------------
        self.ep_2 = c.ep_2 
        self.ep_ini = c.ep_ini
        self.ep_fin = c.ep_fin

        self.eta_fin = c.eta_fin
        self.eta_2 = c.eta_2
        self.eta_ini = c.eta_ini
        self.gamma_wout = c.gamma_wout #0.915 # defaultは0.985最も良い0.925出力層の結合荷重の割引率γ
        self.k_greedy = c.k_greedy # ε-greedy法のεの決定係数

        np.random.seed(c.seed)

        
        #-----------------------------------------------------------------


class HRC_q(Hyperpm_HRC):
    
    def __init__(self,c):
        super().__init__(c)

        # 結合荷重の生成
        self.Wr = generate_random_matrix(self.Nh,self.Nh,self.alpha_r,self.beta_r,distribution="one",normalization="sr")
        self.Wb = generate_random_matrix(self.Nh,self.Ny,self.alpha_b,self.beta_b,distribution="one",normalization="none")
        self.Wi = generate_random_matrix(self.Nh,self.Nu,self.alpha_i,self.beta_i,distribution="one",normalization="none")

        self.Wo = np.zeros(self.Nh * self.Ny).reshape(self.Ny, self.Nh)

        self.Hp = [] 
        self.Hx = [] 
        self.Hs = [] 
        
        self.Yp = [] # RCの出力ypを格納
        #self.Yx = []
        self.Ys = []

        self.Us = []
        self.Up = []
        self.Rs = []

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
        self.epsd_time = 0
        self.success_ratio = 0

        # hrc初期設定-----------------
        self.rs = 1
        #-----------------------------
        self.cnt_overflow = 0


    def initialize_setting(self, actions, epsd_num, learning_mode):
        self.all_actions = actions
        self.all_epsd = epsd_num
        self.learning_mode = learning_mode


    def eta(self,rewards):
        # if self.eta_choice <= 1: # 出力層の結合荷重更新の学習率η(急激に減少)
        #     if self.goal_num <= self.eta_max_goal: #ゴール数により変化する
        #         eta_value =  self.eta_init
        #     else:
        #         eta_value = self.eta_finish
        # elif self.eta_choice == 2:# 出力層の結合荷重更新の学習率η(なめらかに減少)
        #     eta_value =(1/np.exp(self.episode_num/self.t_eta))*self.eta_init
        # elif self.eta_choice >= 3:# 出力層の結合荷重更新の学習率η(指定ステップ後なめらかに減少)
        #     if self.goal_num <= self.eta_max_goal:
        #         self.begin_cnt_eta = self.episode_num
        #         eta_value = self.eta_init
        #     else:
        #         eta_value=(1/np.exp((self.episode_num-self.begin_cnt_eta)/self.t_eta))*self.eta_init
        
        # if self.pre_episode_eta != self.episode_num:
        #     self.Eta_collect.append(eta_value)
        
       

        
        


        recent_reward = np.sum(np.heaviside(np.array(rewards[-20:]),0))/20
        
        eta = self.eta_fin + (self.eta_ini - self.eta_fin) * np.exp(-recent_reward/self.eta_2)

        
        if self.pre_episode_eta != self.episode_num:
            self.Eta_collect.append(eta)
        self.pre_episode_eta = self.episode_num
        return eta

    
    def fy(self, h):
        return np.tanh(h)
    def fyi(self, h):
        return np.arctanh(h)

    def p2s(self, theta, p):
        return np.heaviside( np.sin(np.pi*(2*theta-p)),1)
 

    # ε-greedy法を用いた戦略
    def policy_epsilon(self,rewards):
        # epsilon = max(1- (self.all_stp_num/5000),0)
        # #print(epsilon)
        # epsilon = 0
        # epsilon = self.ep_fin+(self.ep_ini-self.ep_fin)*np.exp(-self.success_ratio/self.ep_2)
        # if self.pre_episode_plcy != self.episode_num:
        #     self.Epsilon_collect.append(epsilon)

        # if np.random.random() < epsilon:
        #     next_action =  np.random.randint(len(self.all_actions))
        # else:
        #     if sum(self.yp) != 0:
        #         next_action = np.argmax(self.yp)
                
        #     else:
        #         next_action =  np.random.randint(len(self.all_actions))
        
        # self.pre_episode_plcy = self.episode_num
        

        recent_reward = np.sum(np.heaviside(np.array(rewards[-20:]),0))/20

        ep = self.ep_fin + (self.ep_ini - self.ep_fin) * np.exp(-recent_reward/self.ep_2)
       

        if np.random.random() < ep:
            next_action =  np.random.randint(len(self.all_actions))
        else:
            if sum(self.yp) != 0:
                next_action = np.argmax(self.yp)
            else:
                next_action =  np.random.randint(len(self.all_actions))
        
        return next_action


    # 1episode終わるごとにネットワークをリセットする
    def reset_network(self, current_episode, current_goal_num):
        self.one_epsd_step_num = 0
        self.episode_num = current_episode
        self.goal_num = current_goal_num

        #neuron
        self.hsign = np.zeros(self.Nh)
        #hx = np.zeros(Nh)
        #self.hx = np.random.uniform(0,1,self.Nh) # [0,1]の連続値
        self.hx = np.zeros(self.Nh) # [0,1]の連続値
        self.hs = np.zeros(self.Nh) # {0,1}の２値
        self.hs_prev = np.zeros(self.Nh)
        self.hc = np.zeros(self.Nh) # ref.clockに対する位相差を求めるためのカウント
        self.hp = np.zeros(self.Nh) # [-1,1]の連続値
        self.hp_pre = np.zeros(self.Nh) # [-1,1]の連続値
        self.ht = np.zeros(self.Nh) # {0,1}

        # output
        #ysign = np.zeros(Ny)
        self.yp = np.zeros(self.Ny) #RCの出力
        self.yp_pre = np.zeros(self.Ny) #RCの出力
        self.yx = np.zeros(self.Ny)
        self.ys = np.zeros(self.Ny)
        #yc = np.zeros(Ny)
        self.epsd_time = 0
        self.prev_hp = 0


     # 行動価値観数の計算
    def step_network_q(self, state):
        # 入力を状態のindexが1のOne-hotベクトルに変換
        change_input = np.zeros(self.Nu) #:TODO
        change_input[state] = 1    
        up = self.fy(change_input) # hrcへの入力

        # any_hs_change = True
        self.ht = 2*self.hs-1 #リファレンスクロック同期用ラッチ動作
        # サイクル(連続時間)-------------------------------------------------------------------------
        #print(state)
        for n in range(self.NN):
            
            #theta = np.mod(self.epsd_time/self.NN,1) # 連続時間t(0,1)
            theta = np.mod(n/self.NN,1) # 連続時間t(0,1)同じ
            self.rs_prev = self.rs
            self.hs_prev = self.hs# 前のニューロンの状態

            self.rs = self.p2s(theta,0)# 参照クロック
            us =    self.p2s(theta,up) # エンコードされた入力(upはfor文内では不変)
            self.ys = self.p2s(theta, self.yp)# 出力をエンコード

            sum = np.zeros(self.Nh)
            sum += self.alpha_s*(self.hs-self.rs)*self.ht # ref.clockと同期させるための結合
            sum += self.Wi@(2*us-1) # 外部入力
            sum += self.Wr@(2*self.hs-1) # リカレント結合
            #sum += self.Wr@(2*(self.p2s(theta,self.hp))-1)

            #if mode == 0:
            #    sum += Wb@ys
            #if mode == 1:  # teacher forcing
            #    sum += Wb@ds

            self.hsign = 1 - 2*self.hs
            self.hx = self.hx + self.hsign*(1.0+np.exp(self.hsign*sum/self.Temp))*self.dt# 連続値である状態
            self.hs = np.heaviside(self.hx+self.hs-1,0)# 離散値である状態
            self.hx = np.fmin(np.fmax(self.hx,0),1)# ?

            # if self.rs==1:
            #     self.hc += self.hs # デコードのためのカウンタ、ref.clockとhsのANDでカウントアップ
            # for i in range(self.Nh):
            #     if self.hs_prev[i]==1 and self.hs[i]==0:
            #         self.hc[i]=self.count
            # self.count += 1
            self.hc[(self.hs_prev==1) & (self.hs==0)] = n # hs の立ち下がりの時間を hc に保持する。
            

            # ref.clockの立ち上がり
            # if self.rs_prev==0 and self.rs==1:
            #     self.hp = 2*self.hc/self.NN-1 # デコード、カウンタの値を連続値に変換
            #     self.hc = np.zeros(self.Nh) #カウンタをリセット
            #     self.ht = 2*self.hs-1 #リファレンスクロック同期用ラッチ動作
            #     self.yp = self.fy(self.Wo@self.hp)
            #     self.count = 0

            #     # record-------
            #     if self.learning_mode == 1 and self.episode_num >= self.all_epsd-30:
            #         self.Us.append(us)
            #         self.Hp.append(self.hp)
            #         self.Yp.append(self.yp)
            #     #--------------

            any_hs_change = np.any(self.hs!=self.hs_prev)
            

            # record------------------
            if self.learning_mode == 1 and self.episode_num >= self.all_epsd-2:
                self.Us.append(us)
                self.Rs.append(self.rs)
                self.Hx.append(self.hx)
                self.Hs.append(self.hs)
                #self.Yx.append(self.yx)
                self.Ys.append(self.ys)
            #------------------------

            self.epsd_time += 1
        #--------------------------------------------------------------------------------------------
        

        # 離散時間(サイクル終了)
        self.hp = 2*self.hc/self.NN-1 # デコード、カウンタの値を連続値に変換
        if 2 <= self.epsd_time < self.all_epsd-1:
            tmp = np.sum( np.heaviside( np.fabs(self.hp-self.prev_hp) - 0.6 ,0))
            self.cnt_overflow += tmp
        
        self.prev_hp = self.hp


        self.hc = np.zeros(self.Nh) # カウンタをリセット
        
        self.yp = self.fy(self.Wo@self.hp)# 出力

        
        
        #print(tmp)

        # record------------------------------------------------------------
        if self.learning_mode == 1 and self.episode_num >= self.all_epsd-20:
            self.Up.append(up)
            self.Hp.append(self.hp)
            self.Yp.append(self.yp)
        #-------------------------------------------------------------------

        self.one_epsd_step_num += 1 #:TODO
        self.all_stp_num += 1 #:TODO


    ### training output conection
    def train_weight_q(self, action, reward,rewards):
        Wo_next = self.Wo
        Wo_next[action] = self.Wo[action]+self.eta(rewards)*\
            np.tanh(reward+self.gamma_wout*max(self.yp)-self.yp_pre[action])*self.hp_pre

        ### Update
        self.Wo = Wo_next
        self.yp_pre = self.yp
        self.hp_pre = self.hp


    ### Update
    def hrc_update_q(self):
        self.yp_pre = self.yp
        self.hp_pre = self.hp
        

    def plot_hrc_episode(self):
        fig = plt.figure(figsize=(20, 12))
        Nr = 3
        ax = fig.add_subplot(Nr,1,1)
        ax.cla()
        ax.set_title("Up")
        ax.plot(self.Up)

        ax = fig.add_subplot(Nr,1,2)
        ax.cla()
        ax.set_title("Hp")
        ax.plot(self.Hp)

        ax = fig.add_subplot(Nr,1,3)
        ax.cla()
        ax.set_title("Yp")
        ax.plot(self.Yp)

        plt.show()


    def plot_hrc_step(self):
        fig = plt.figure(figsize=(20, 12))
        Nr = 4

        ax = fig.add_subplot(Nr,1,1)
        ax.cla()
        ax.set_title("Us")
        ax.plot(self.Us)

        ax = fig.add_subplot(Nr,1,2)
        ax.cla()
        ax.set_title("Rs")
        ax.plot(self.Rs)

        ax = fig.add_subplot(Nr,1,3)
        ax.cla()
        ax.set_title("Hx")
        ax.plot(self.Hx)

        ax = fig.add_subplot(Nr,1,4)
        ax.cla()
        ax.set_title("Hs")
        ax.plot(self.Hs)

        plt.show()


    def plot_hrc_out(self):
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
        ax.set_title("Yp")
        ax.plot(self.Yp)

        plt.show()