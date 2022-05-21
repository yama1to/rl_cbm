# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# ロボット制御用レザバー
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import hippocampal
#hc  = hippocampal.HippocampalCell()
import cupy as cp 
class Agent:
    NN = 256
    Temp=1


    Nu = 8 #129= 121+8 #size of input
    Nx = 500 #size of dynamical reservior
    Ny = 3   #size of output

    sigma_np = -5
    alpha_r = 0.9
    alpha_b = 0.5#0.8
    alpha_i = 0.5
    alpha_o = 0.8
    alpha_s = 1

    beta_r = 0.1
    beta_b = 0.05
    beta_i = 0.05
    alpha0 = 1.0
    beta = 1.0
    gamma = 0.90

    alpha_P = 0.02

    tau_x = 2.0
    lambda0 = 0.1

    tau_s = 20
    sigma_init = 0.05
    sigma_final = 0.001 #0.02
    tau_sigma = 2000

    eta_init = 0.02 # 0.0005
    eta_final = 0.001
    tau_eta = 1000

    cnt_goal1 = 0
    goal_s = 0
    episode = 0

    def __init__(self,config=None,plot=1):
        c = config
        self.plot = plot
        self.alpha_r = 0.3
        self.alpha_b = 0.3#0.8
        self.alpha_i = 0.3
        self.alpha_o = 0.3
        self.alpha_s = 1

        self.beta_r = 0.1
        self.beta_b = 0.05
        self.beta_i = 0.05

        if not c == None:
            self.alpha_r = c.alpha_r
            self.alpha_b = c.alpha_b
            self.alpha_i = c.alpha_i
            self.alpha_o = c.alpha_o
            self.alpha_s = c.alpha_s

            self.beta_r = c.beta_r
            self.beta_b = c.beta_b
            self.beta_i = c.beta_i

        return None

    def eta(self):
        eta = self.eta_final + (self.eta_init-self.eta_final) * np.exp(-self.t/self.tau_eta)
        return eta

    def eta2(self):
        """
        self.mean_reward:一定エピソードの報酬平均
        報酬平均に応じて学習率を変化させる。
        段階的に変化させているが、関数を使って連続的に変化させると望ましい気がする
        """
        if self.mean_reward >= 7:
            eta = 0
        elif self.mean_reward >=5 and self.mean_reward <7:
            eta = 0.0001
        elif self.mean_reward >=3 and self.mean_reward <5:
            eta = 0.001
        elif self.mean_reward >=1 and self.mean_reward <3:
            eta = 0.01
        else:
            eta = 0.01
        return eta

    def sigma_s(self):
        return self.sigma_final + (self.sigma_init-self.eta_final) * cp.exp(-self.t/self.tau_sigma)

    def initialize(self):
        self.n = 0 #　resetによって0になる
        self.t = 0 # resetでは初期化されない。
        self.generate_weight_matrix()
    
    def ring_weight(self,):
        Wr = cp.zeros((self.Nx,self.Nx))
        for i in range(self.Nx-1):
            Wr[i,i+1] = 1
        Wr[-1,0]=1
        v = cp.linalg.eigvals(Wr)
        lambda_max = max(abs(v))
        Wr = Wr/lambda_max*self.alpha_r
        return Wr 



    def generate_weight_matrix(self):
        ### NOTE:ローカル変数を定義する。「self.」と書くと長くなるので。
        Nx = self.Nx
        Ny = self.Ny
        Nu = self.Nu

        ### Wr
        Wr0 = np.zeros(Nx * Nx)
        nonzeros = Nx * Nx * self.beta_r
        Wr0[0:int(nonzeros / 2)] = 1
        Wr0[int(nonzeros / 2):int(nonzeros)] = -1
        np.random.shuffle(Wr0)
        Wr0 = Wr0.reshape((Nx, Nx))
        v = scipy.linalg.eigvals(Wr0)
        lambda_max = max(abs(v))
        self.Wr = cp.asarray(Wr0 / lambda_max * self.alpha_r)
        #self.Wr = self.ring_weight()

        # print("lamda_max",lambda_max)
        # print("Wr:")
        # print(Wr)

        ### Wb
        Wb0 = np.zeros(Nx * Ny)
        Wb0[0:int(Nx * Ny * self.beta_b / 2)] = 1
        Wb0[int(Nx * Ny * self.beta_b / 2):int(Nx * Ny * self.beta_b)] = -1
        np.random.shuffle(Wb0)
        Wb0 = Wb0.reshape((Nx, Ny))
        self.Wb = cp.asarray(Wb0 * self.alpha_b)

        # print("Wb:")
        # print(Wb)

        ### Wi
        Wi0 = np.zeros(Nx * Nu)
        Wi0[0:int(Nx * Nu * self.beta_i / 2)] = 1
        Wi0[int(Nx * Nu * self.beta_i / 2):int(Nx * Nu * self.beta_i)] = -1
        np.random.shuffle(Wi0)
        Wi0 = Wi0.reshape((Nx, Nu))
        self.Wi = cp.asarray(Wi0 * self.alpha_i)
        # print("Wi:")

        ### Wo

        self.Wo = cp.zeros((Ny,Nx))

    def reset_network(self,episode,mean_reward):
        self.n = 0#time step
        self.episode = episode
        #neuron

        self.hx = cp.zeros(self.Nx)
        self.hs = cp.zeros(self.Nx) # {0,1}の２値
        self.hs_prev = cp.zeros(self.Nx)
        
        self.x = cp.zeros(self.Nx)
        #self.x = np.random.uniform(-1, 1, self.Nx) * 0.1

        self.r = cp.tanh(self.x)
        self.s = cp.zeros(self.Ny)

        #output
        self.q = cp.zeros(self.Ny)
        self.P = cp.identity(self.Nx)/self.alpha_P
        self.a = cp.random.randint(0,3)

        #check
        self.mean_reward = mean_reward#一定エピソードの報酬平均

        #save
        self.Hx = []
        self.Hs = []
        self.R = []
        self.Q = []
        self.U = []

    def get_action(self,state, reward):
        ### 感覚情報についての前処理を行う。

        u1 = cp.asarray(state[ :8])# 距離センサー
        u = cp.exp(-u1/100)
        action = self.step_network(u,reward)

        return cp.asnumpy(action)
    
    def p2s(self,theta,p):
        tmp =cp.sin(cp.pi*(2*theta-p))
        tmp[tmp>=0] = 1
        tmp[tmp<0]  = 0
        return tmp

    def step_network(self, state, reward):
        u = state#input
        reward = cp.asarray(reward)
        dt = 1.
        
        hc = cp.zeros(self.Nx)
        
        ht = 2*self.hs-1
        

        for n in range(self.NN):
            theta = cp.mod(n/self.NN,1)
            rs = self.p2s(theta,0)# 参照クロック
            us = self.p2s(theta,u) # エンコードされた入力
            #hs = 
            qs = self.p2s(theta,self.q)
            
            sum = cp.zeros(self.Nx)
            sum += self.Wi @ us
            #sum += self.Wr @ (2*self.p2s(theta,self.r)-1)
            sum += self.Wr @ self.hs
            sum += self.Wb @ qs
            sum += self.alpha_s *(self.hs-rs)*ht

            hsign = 1 - 2*self.hs
            self.hx = self.hx + hsign*(1.0+cp.exp(hsign*sum/self.Temp))/self.NN
            self.hs[self.hx+self.hs-1>=0] = 1
            self.hs[self.hx+self.hs-1<0]  = 0


            self.hx = cp.fmin(np.fmax(self.hx,0),1)

            hc[(self.hs_prev == 1)& (self.hs==0)] = n 

            if self.plot : 
                self.Hx.append(self.hx)
                self.Hs.append(self.hs)

        r_next = 2 * hc/self.NN - 1

        q_next = self.Wo @ r_next
        s_next = (1 -dt /self.tau_s)*self.s + self.sigma_s()*cp.random.normal(0,1,self.Ny)
        qt_next = q_next + s_next
        a_next = cp.argmax(qt_next)

        ### training output conection
        Wo_next = self.Wo
        Wo_next[self.a] = self.Wo[self.a] + self.eta2()*cp.tanh(reward +self.gamma*qt_next[a_next]-self.q[self.a])*self.r#変更

        ### epsilon greedy
        epsilon = 0.2/(1 + self.mean_reward/3)#変更、報酬平均に応じて変化
        if epsilon > cp.random.uniform(0, 1):
            a_next = cp.random.choice([0,1,2],1)


        ### Update
        self.q = q_next
        self.r = r_next
        self.s = s_next
        self.Wo = Wo_next
        self.a = a_next

        self.n += 1
        self.t += 1
        #print(self.q)

        ### Record

        if self.plot:
            #self.X.append(self.x.reshape(1,self.Nx))

            self.R.append(self.r)
            self.Q.append(self.q)
            self.U.append(u)

        return cp.asnumpy(self.a)
