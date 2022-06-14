# coding: utf-8
# Copyright (c) 2017-2020 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# ロボット制御用レザバー

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import maincbm


class Agent:
    def __init__(self,c):
    ### node ###

        self.plot = c.plot
        self.NN = c.NN

        self.Nu = c.Nu #size of input
        self.Nu2 = c.Nu2 #size of input signal
        self.Nx = c.Nx#size of dynamical reservior
        self.Ny = c.Ny #size of output

        ### Hyperparameter ##
        # weight #
        self.alpha_r = c.alpha_r #Spectral radius
        self.beta_r = c.beta_r #Bond density
        self.alpha_i = c.alpha_i #Input weight
        self.beta_i = c.beta_i  #Bond density
        self.gamma = c.gamma #減衰率　
        self.tau_x = c.tau_x #時定数

        self.alpha_s = c.alpha_s
        self.Temp = c.Temp

        # 学習率(Step size parameter) #
        self.eta_2 = c.eta_2
        self.eta_ini = c.eta_ini
        self.eta_fin = c.eta_fin
        
        # 探索(epsilon) #
        self.ep_2 = c.ep_2
        self.ep_ini = c.ep_ini
        self.ep_fin = c.ep_fin

        # other #
        self.alpha_x = c.alpha_x
        self.alpha0 = c.alpha0
        self.beta = c.beta

        
        self.now_epi = c.now_epi
        self.pre_epi = 0

        self.Eta = []

        #self.sigma_np = -5
        # ### node ###
        # self.plot = 1
        # self.NN = 256


        # self.Nu = 8 #size of input
        # self.Nu2 = 2 #size of input signal
        # self.Nx = 1000#size of dynamical reservior
        # self.Ny = 3 #size of output

        # ### Hyperparameter ###
        # # weight #
        # self.alpha_r = 0.94 #Spectral radius
        # self.beta_r = 0.3 #Bond density
        # self.alpha_i = 0.8 #Input weight
        # self.beta_i = 0.3  #Bond density
        # self.gamma = 0.95 #減衰率　
        # self.tau_x = 6 #時定数

        # self.alpha_s = 1
        # self.Temp = 1

        # # 学習率(Step size parameter) #
        # self.eta_2 = 0.18
        # self.eta_ini = 0.003
        # self.eta_fin = 0
        
        # # 探索(epsilon) #
        # self.ep_2 = 0.17
        # self.ep_ini = 0.62
        # self.ep_fin = 0

        # # other #
        # self.alpha_x = 1.0
        # self.alpha0 = 1.0
        # self.beta = 1.0
        # #self.sigma_np = -5


    ### 初期設定 ###
    def initialize(self):
        self.n = 0 #ステップ時間
        self.t = 0 #全てのエピソード時間(resetされない)
        self.success_ratio = 0#学習率やεを決定する指標(成功確率)
        self.X_pca_right = [] #pca
        self.X_pca_left = [] #pca
        self.generate_weight_matrix()#ネットワーク

    def ring_weight(self,):
        Wr = np.zeros((self.Nx,self.Nx))
        for i in range(self.Nx-1):
            Wr[i,i+1] = 1
        Wr[-1,0]=1
        v = np.linalg.eigvals(Wr)
        lambda_max = max(abs(v))
        Wr = Wr/lambda_max*self.alpha_r
        return Wr 

    def rec_weight(self,Nx):
        Wr0 = np.zeros(Nx * Nx)
        nonzeros = Nx * Nx * self.beta_r
        Wr0[0:int(nonzeros / 2)] = 1
        Wr0[int(nonzeros / 2):int(nonzeros)] = -1
        np.random.shuffle(Wr0)
        Wr0 = Wr0.reshape((Nx, Nx))
        v = scipy.linalg.eigvals(Wr0)#固有値をもとめている
        lambda_max = max(abs(v))
        Wr = Wr0 / lambda_max * self.alpha_r
        return Wr
    
    def in_weight(self,Nx,Nu):
        Wi0 = np.zeros(Nx * Nu)
        Wi0[0:int(Nx * Nu * self.beta_i / 2)] = 1
        Wi0[int(Nx * Nu * self.beta_i / 2):int(Nx * Nu * self.beta_i)] = -1
        np.random.shuffle(Wi0)
        Wi0 = Wi0.reshape((Nx, Nu))
        Wi = Wi0 * self.alpha_i
        return Wi 

    ### ネットワーク ###
    def generate_weight_matrix(self):
        # NOTE:ローカル変数を定義する。「self.」と書くと長くなるので。
        Nx = self.Nx
        Ny = self.Ny

        # # Wr　中間層 #

        self.Wr = self.rec_weight(self.Nx)
        #self.Wr = self.ring_weight()


        # Wb出力層から中間層へのフィードバック #
        '''
        Wb0 = np.zeros(Nx * Ny)
        Wb0[0:int(Nx * Ny * self.beta_b / 2)] = 1
        Wb0[int(Nx * Ny * self.beta_b / 2):int(Nx * Ny * self.beta_b)] = -1
        np.random.shuffle(Wb0)
        Wb0 = Wb0.reshape((Nx, Ny))
        self.Wb = Wb0 * self.alpha_b#
        '''
        # Wi　入力層から中間層 #
        self.Wi = self.in_weight(self.Nx,self.Nu)
        
        # Wi　入力層から中間層2(信号．重みは同じ) #
        self.Wi2 = self.in_weight(self.Nx,self.Nu2)

        # 出力 #
        self.Wo = np.zeros((Ny,Nx))

    
    ### 初期化(エピソードの初めに) ###
    def reset_network(self,episode,success):
        # internal #
        self.hx = np.zeros(self.Nx)
        self.hs = np.zeros(self.Nx) # {0,1}の２値
        self.hs_prev = np.zeros(self.Nx)
        self.x = np.zeros(self.Nx)
        self.r = np.tanh(self.x)
        self.s = np.zeros(self.Ny)
        self.q = np.zeros(self.Ny)
        # action #
        self.a = np.random.randint(0,3)
        # other #
        self.n = 0 #time step
        self.count_signal = 0
        self.episode = episode

        
        # 報酬の獲得率を計算 #
        if (self.episode+1)%20 == 0:
            self.success_ratio = sum(success[self.episode-21:self.episode-1])/20
        # save #v
        if self.plot:
            self.Hx = []
            self.Hs = []
            self.X = []
            self.R = []
            self.Q = []
            self.U = []
    
            
    ### 行動 ###
    def get_action(self,state,reward,requirement):
        """
        信号の確認:requirement[signal_right, signal_left, signal_check]
        """
        u1 = state[ :8]#距離センサー
        u = np.exp(-u1/100)#距離センサの値を変形
        action = self.step_network(u,reward,requirement)#学習の更新
        return action

    def p2s(self,theta,p):
        return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

    ### 学習の更新 ###
    def step_network(self,state,reward,requirement):
        dt = 1.0
        u = state #input
        hc = np.zeros(self.Nx)
        ht = 2*self.hs-1

        u2 = np.array([0,0]) #input singal
        if self.plot:
            Hx_append = self.Hx.append
            Hs_append = self.Hs.append
        
        # 信号 #
        if requirement[2]:#信号線を超えた時
            self.count_signal = 4
            
        if self.count_signal > 0:#複数ステップ与える
            u2 = np.tanh(np.array([requirement[0],requirement[1]]))#requirment=[信号右、信号左]
            self.count_signal -= 1

        
        for n in range(self.NN):
            self.hs_prev = self.hs
            theta = np.mod(n/self.NN,1)
            #print(self.p2s(theta,1))
            
            rs = self.p2s(theta,0)# 参照クロック
            us = self.p2s(theta,u) # エンコードされた入力
            us2 = self.p2s(theta,u2) # エンコードされた入力


            #qs = self.p2s(theta,self.q)
            
            sum = np.zeros(self.Nx)
            sum += self.Wi @ (2*us-1)
            sum += self.Wi2 @ (2*us2-1)
            #sum += self.Wr @ (2*self.p2s(theta,self.r)-1)
            #print(sum.shape, self.Wr.shape, self.hs.shape,)
            sum += self.Wr @ (2*self.hs-1)
            #sum += self.Wb @ qs
            sum += self.alpha_s *(self.hs-rs)*ht

            hsign = 1 - 2*self.hs
            self.hx = self.hx + hsign*(1.0+np.exp(hsign*sum/self.Temp))/self.NN
            self.hs = np.heaviside(self.hx+self.hs-1,0)
            self.hx = np.fmin(np.fmax(self.hx,0),1)

            hc[(self.hs_prev == 1)& (self.hs==0)] = n 

  
            if self.plot : 
                Hx_append(self.hx)
                Hs_append(self.hs)

        r_next = 2 * hc/self.NN - 1

        # 出力層の状態 #
        q_next = self.Wo @ r_next
        qt_next = q_next
        a_next = np.argmax(qt_next)

        # 学習 #
        eta = self.eta_fin+(self.eta_ini-self.eta_fin)*np.exp(-self.success_ratio/self.eta_2)
        if self.now_epi != self.pre_epi:
            self.Eta.append(eta)

        self.pre_epi = self.now_epi


        # if self.success_ratio > 0.8:
        #     eta = 0

        Wo_next = self.Wo.copy()
        Wo_next[self.a] = self.Wo[self.a] + eta*(reward  + self.gamma*qt_next[a_next]-self.q[self.a])*self.r#np.tannなし

        # 行動 #
        """
        epsilonはゴールの到達率によって変化
        0から1のランダム数が、epsilon以下のときに行動はランダムな動作となる
        """
        epsilon = self.ep_fin+(self.ep_ini-self.ep_fin)*np.exp(-self.success_ratio/self.ep_2)
        # if self.success_ratio > 0.8:
        #     epsilon = 0

        if epsilon > np.random.uniform(0, 1):
            a_next = np.random.choice([0,1,2])#ランダムな行動
            
        if self.count_signal > 0:#信号が出ているときは直進のみ選択する
            a_next = 2#forward
        
        # update #

        self.q = q_next
        self.a = a_next
        self.r = r_next
        self.Wo = Wo_next
        self.n += 1#タイムステップ
        self.t += 1#全エピソード時間

        # save #
        u3 = np.concatenate([u,u2])#センサ値と信号
        
        if self.plot:
            self.R.append(self.r)
            self.Q.append(self.q)
            #self.U = np.append(self.U,u.reshape(1,self.Nu), axis=0)
            self.U.append(u)
        
        if self.episode > maincbm.Config().num_episodes-50:#pca
            # 信号に応じて異なるリストへ入れる #
            if requirement[0] == 1:
                self.X_pca_right.append(self.r)
            if requirement[1] == 1:
                self.X_pca_left.append(self.r)
            
        return a_next

        