# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# ロボット制御用レザバー
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import hippocampal
#hc  = hippocampal.HippocampalCell()

class Agent:
    Nu = 8 #129= 121+8 #size of input
    Nx = 1200 #size of dynamical reservior
    Ny = 3   #size of output

    sigma_np = -5
    alpha_r = 0.8
    alpha_b = 0.#0.8
    alpha_i = 0.5
    alpha_o = 0.8
    beta_r = 0.05
    beta_b = 0.0
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

    def __init__(self):
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
        return self.sigma_final + (self.sigma_init-self.eta_final) * np.exp(-self.t/self.tau_sigma)

    def initialize(self):
        self.n = 0 #　resetによって0になる
        self.t = 0 # resetでは初期化されない。
        self.generate_weight_matrix()

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
        self.Wr = Wr0 / lambda_max * self.alpha_r

        # print("lamda_max",lambda_max)
        # print("Wr:")
        # print(Wr)

        ### Wb
        Wb0 = np.zeros(Nx * Ny)
        Wb0[0:int(Nx * Ny * self.beta_b / 2)] = 1
        Wb0[int(Nx * Ny * self.beta_b / 2):int(Nx * Ny * self.beta_b)] = -1
        np.random.shuffle(Wb0)
        Wb0 = Wb0.reshape((Nx, Ny))
        self.Wb = Wb0 * self.alpha_b
        # print("Wb:")
        # print(Wb)

        ### Wi
        Wi0 = np.zeros(Nx * Nu)
        Wi0[0:int(Nx * Nu * self.beta_i / 2)] = 1
        Wi0[int(Nx * Nu * self.beta_i / 2):int(Nx * Nu * self.beta_i)] = -1
        np.random.shuffle(Wi0)
        Wi0 = Wi0.reshape((Nx, Nu))
        self.Wi = Wi0 * self.alpha_i
        # print("Wi:")

        ### Wo
        #Wo = np.ones(Ny * Nx)
        #Wo = Wo.reshape((Ny, Nx))
        #self.Wo = Wo
        #self.Wo = np.ones(Ny * Nx).reshape((Ny, Nx))
        # print(Wo)

        #tmp = np.sqrt(1 / Nx)
        #Wo0 = np.random.normal(0, 1, Ny * Nx) * tmp * self.alpha_o
        #np.random.shuffle(Wo0)
        #self.Wo = Wo0.reshape((Ny, Nx))
        self.Wo = np.zeros((Ny,Nx))
        #Wo = np.random.uniform(-1, 1, Nz * Nx)
        #Wo = Wo.reshape((Nz, Nx))

    def reset_network(self,episode,mean_reward):
        self.n = 0#time step
        self.episode = episode
        #neuron
        #self.firing_rate = np.zeros(self.Nx)
        #self.resource_sig = np.zeros(self.Nx)
        #self.utili_para = np.zeros(self.Nx)
        self.x = np.zeros(self.Nx)
        #self.x = np.random.uniform(-1, 1, self.Nx) * 0.1

        self.r = np.tanh(self.x)
        self.s = np.zeros(self.Ny)

        #output
        self.q = np.zeros(self.Ny)
        self.P = np.identity(self.Nx)/self.alpha_P
        self.a = np.random.randint(0,3)

        #check
        self.mean_reward = mean_reward#一定エピソードの報酬平均

        #save
        self.X = []
        self.R = []
        self.Q = []
        self.U = []

    def get_action(self,state, reward):
        ### 感覚情報についての前処理を行う。

        u1 = state[ :8]# 距離センサー
        #vp = hc.encode_place(state[9],state[10])# ロボットの位置座標を海馬場所細胞に変換
        #vp_object = env.object_area
        #pre_action = res.y
        #u = u1/300
        #u = u1
        #u = 1/(1+u1/30)
        u = np.exp(-u1/100)
        #dis_goal = state[13]
        #print(dis_goal,1.0/(1+dis_goal/100))
        #print("u1:",u1)
        #u = np.append(u,1.0/(1+dis_goal/100))

        #u = np.append(vp,u1)
        #u = np.append(u,vp_object)
        #u = np.append(u,pre_action)

        action = self.step_network(u,reward)
        #action = res.step_network_synapses(u,reward)
        #action = res.step_network_P(u,reward)
        #print(res.y)

        return action

    def step_network(self, state, reward):
        u = state#input
        dt = 1.0
        sum = np.zeros(self.Nx)
        sum += self.Wi @ u
        sum += self.Wr @ self.r
        #sum += self.Wb @ self.q
        x_next = (1 -dt /self.tau_x)*self.x +dt/self.tau_x*(sum)
        r_next = np.tanh(self.beta*x_next)
        q_next = self.Wo @ r_next
        s_next = (1 -dt /self.tau_s)*self.s + self.sigma_s()*np.random.normal(0,1,self.Ny)
        qt_next = q_next + s_next
        a_next = np.argmax(qt_next)

        ### training output conection
        Wo_next = self.Wo
        #Wo_next[a_next] = self.Wo[a_next] + self.eta()*np.tanh(reward +self.gamma*qt_next[a_next]-self.q[a_next])*r_next
        Wo_next[self.a] = self.Wo[self.a] + self.eta2()*np.tanh(reward +self.gamma*qt_next[a_next]-self.q[self.a])*self.r#変更

        ### epsilon greedy
        #epsilon = 0.0/(1 + self.episode/100)#self.epsilon2(self.reward_split)
        epsilon = 0.2/(1 + self.mean_reward/3)#変更、報酬平均に応じて変化
        if epsilon > np.random.uniform(0, 1):
            a_next = np.random.choice([0,1,2])

        #print(self.n,reward,action,q_next,s_next)
        #print("%4d %5.2f %d %s %s" % (self.n,reward,action,q_next,s_next))

        ### Update
        self.x = x_next
        self.q = q_next
        self.r = r_next
        self.s = s_next
        self.Wo = Wo_next
        self.a = a_next

        self.n += 1
        self.t += 1
        #print(self.q)

        ### Record
        self.X.append(self.x)
        self.R.append(self.r)
        self.Q.append(self.q)
        self.U.append(u)

        return self.a
