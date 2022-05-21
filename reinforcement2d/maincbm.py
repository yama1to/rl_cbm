# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)

import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import sys
import agentcbm
import environment

from tqdm import tqdm 

import argparse
from explorer import common 


import warnings
warnings.simplefilter('ignore')


class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0# 図の出力のオンオフ

        # config
        self.dataset=6
        self.seed:int=220 # 乱数生成のためのシード
        self.num_episodes =600#エピソードの回数
        self.test_episodes = 50

        self.max_number_of_steps = 200#最大ステップ

        self.render =0
        self.verbose = 0

        self.Temp=1

        self.Nu = 8 #129= 121+8 #size of input
        self.Nx = 500 #size of dynamical reservior
        self.Ny = 3   #size of output

        self.sigma_np = -5
        self.NN = 2**8
        

        self.alpha_i = 0.52
        self.alpha_r = 0.41
        self.alpha_b = 0.
        self.alpha_s = 0.92
        self.alpha_o = 0.01

        self.beta_i = 0.52
        self.beta_r = 0.1
        self.beta_b = 0.
        self.beta_o = 0.1
        self.alpha0 = 1.0
        self.beta = 1.0
        self.gamma = 0.90

        self.alpha_P = 0.02

        self.tau_x = 2.0
        self.lambda0 = 0.1

        self.tau_s = 20
        self.sigma_init = 0.005
        self.sigma_final = 0.001 #0.02
        self.tau_sigma = 2000

        self.eta_init = 0.02 # 0.0005
        self.eta_final = 0.001
        self.tau_eta = 1000


        # ResultsX
        self.cnt_overflow = 0
        self.plus_reward_times=0
        self.mean_reward = 0
        self.new_mean_reward = 0
        self.sum1_reward = 0
        self.sum2_reward = 0
        self.cnt_goal1 = 0
        self.goal_s = 0
        self.episode =0

def get_action4():#ジョイスティックまたはキーボードでの操作
    a = np.zeros(2)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    ### ジョイスティックがあれば使用する。無ければキーボードを使用する。
    try:# ジョイスティック
        joystick
        pressed_keys = pygame.key.get_pressed()
        a[0] = joystick.get_axis(0)# 回転
        a[1] =-joystick.get_axis(1)# 前後
    except:# キーボード
        pressed_keys = pygame.key.get_pressed()
        #if pressed_keys[K_ESCAPE]:sys.exit()
        if pressed_keys[K_LEFT] :a[0]= 1
        if pressed_keys[K_RIGHT]:a[0]= -1
        if pressed_keys[K_UP]   :a[1]= 1
        if pressed_keys[K_DOWN] :a[1]= -1
    return a

def execute(c):
    np.random.seed(int(c.seed))
    
    #pygame.init()
    agentcbm.initialize()
    total_goal = 0
    sum2_reward = 0
    mean_reward = 0#一定エピソードの報酬平均
    plus_reward_times = []#エピソード報酬を格納
    goal = 0 
    sum1_reward_list = []
    sum1_reward_list_append = sum1_reward_list.append
    train = 1
    for episode in tqdm(range(c.num_episodes)):
        #環境の初期化
        agentcbm.reset_network(episode,mean_reward)
        action = 2
        state, reward, done, info = env.reset()
        state, reward, done, info = env.step(action)
        sum1_reward = 0
        plus_reward_times_append = plus_reward_times.append

        if episode >= c.num_episodes - c.test_episodes:
            train = 0
            

        for t in range(c.max_number_of_steps):
            #action = get_action4()
            #action = agentcbm.get_action(state, reward)
            #state, reward, done, info = env.step(action)
            state, reward, done, GOAL = env.step(action)
            
            action = agentcbm.get_action(state, reward,train = train )
            sum1_reward += reward
            sum2_reward += reward
            
            
            if c.verbose:
                print("ep:%4d t:%4d done:%d r:%5.2f R:%5.2f a:%d q:%s s:%s" %
                (episode,agentcbm.t,done,reward,sum1_reward,action,agentcbm.q,agentcbm.s))

            if c.render == 1:env.render()
            

            #print(GOAL,GOAL==1)
            if GOAL==1:
                done = 1
                #print("here")
                if train:
                    c.cnt_goal1 += GOAL
                else:
                    c.goal_s += GOAL
            
                

            if t == c.max_number_of_steps-1:
                done = 1

            if done == 1:
                #print("ep:%4d sum1_reward:%4f" % (episode,sum1_reward))
                plus_reward_times_append(sum1_reward)
                break

            if 2 <= t < c.max_number_of_steps -1:
                tmp = np.sum( np.heaviside( np.fabs(agentcbm.r-prev_r) - 0.6 ,0))
                c.cnt_overflow += tmp

            prev_r = agentcbm.r

            # for event in pygame.event.get():
            #     if event.type == QUIT:
            #         pygame.quit()
            #         if plot == 1:
            #             plot()
            #         sys.exit()
        #print(c.cnt_goal1)   
        #最新20エピソードの報酬総和の平均
        sum1_reward_list_append(sum1_reward)
        if episode>20:
            mean_reward = sum(sum1_reward_list[episode-19:])/20
            c.new_mean_reward = mean_reward
            #print(mean_reward)


    env.close()
    plus_reward_times = np.array(sum1_reward_list)
    plus_reward_times = np.sum(np.heaviside(plus_reward_times,0))
    
    c.plus_reward_times = int(plus_reward_times)
    c.mean_reward = sum2_reward/c.num_episodes
    c.sum1_reward = sum1_reward
    c.sum2_reward = sum2_reward
    c.new_mean_reward = mean_reward


    c.cnt_overflow = c.cnt_overflow/agentcbm.Nx/len(sum1_reward_list)
    
    #np.save(file='seed='+str(c.seed),arr=sum1_reward_list)
    plt.plot(sum1_reward_list)
    plt.save("savefig"+str(c.seed))
    # from explorer import visualization as vs
    # import pandas as pd 
    # df = pd.DataFrame(data=sum1_reward_list, columns=list(range(len(sum1_reward_list))))
    # x,ymean,ystd,ymin,ymax = vs.analyze(df,"x1","y1")
    # cmap = plt.get_cmap("tab10")
    # plt.errorbar(x,ymean,yerr=ystd,fmt='o',color=cmap(2),capsize=2,label="reward")

    

    print("Times of Plus Rwrd=%d,  Mean Rwrd in All Epi=%.2lf, overflow=%.2lf, \
    Total Rwrd=%d, Mean Rwrd of new 100=%.2lf,Train Goal times=%d/%d, Test Goal times=%d/%d" 
        % (c.plus_reward_times,c.mean_reward,c.cnt_overflow,c.sum2_reward,c.new_mean_reward,c.cnt_goal1,c.num_episodes - c.test_episodes,c.goal_s,c.test_episodes))
    
    if c.plot:
        plot()


def plot():
    fig=plt.figure(figsize=(8,6))#figsize=(8,6)
    ax = fig.add_subplot(5,1,1)
    ax.cla()
    ax.plot(agentcbm.U)
    ax.set_ylabel("U")
    ax.set_ylim(-1,1)

    ax = fig.add_subplot(5,1,2)
    ax.cla()
    ax.plot(agentcbm.Hx)
    ax.set_ylabel("Hx")
    ax.set_ylim(0,1)

    ax = fig.add_subplot(5,1,3)
    ax.cla()
    ax.plot(agentcbm.Hs)
    ax.set_ylabel("Hs")
    ax.set_ylim(0,1)

    ax = fig.add_subplot(5,1,4)
    ax.cla()
    ax.plot(agentcbm.R)
    ax.set_ylabel("R")
    ax.set_ylim(-1,1)

    ax = fig.add_subplot(5,1,5)
    ax.cla()
    ax.plot(agentcbm.Q)
    ax.set_ylabel("Q")
    ax.set_xlabel("n")
    plt.show()
    plt.savefig('./fig_dir/'+common.string_now()+'Nh=%d_episode_')



if __name__=="__main__":
    # if sys.argv:
    #     del sys.argv[1:]

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-config", type=str)
    # a = ap.parse_args()

    # c=Config()
    
    # if a.config: c=common.load_config(a)
    # env = environment.MyRobotEnv(c.render)
    # agentcbm = agentcbm.Agent(c)
    # execute(c)
    # if a.config: common.save_config(c)
    c=Config()
    for i in range(10):
        c.seed = i
        env = environment.MyRobotEnv(c.render)
        agentcbm = agentcbm.Agent(c)
        execute(c)
        plt.show()
        




    
