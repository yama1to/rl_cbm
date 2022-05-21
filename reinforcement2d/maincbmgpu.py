# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)

import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import sys
import agentcbmgpu
import environment

from tqdm import tqdm 

import argparse
from explorer import common 





class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0 # 図の出力のオンオフ
        self.render = 0 #シミュレーション表示

        self.verbose = 0
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=1 # 乱数生成のためのシード
        self.num_episodes = 300#エピソードの回数
        self.max_number_of_steps = 200#最大ステップ

        self.alpha_i = 0.1
        self.alpha_r = 0.3
        self.alpha_b = 0.
        self.alpha_s = 0.3
        self.alpha_o = 0.3

        self.beta_i = 0.1
        self.beta_r = 0.1
        self.beta_b = 0.1
        self.beta_o = 0.1

        # ResultsX
        self.cnt_overflow = 0
        self.plus_reward_times=0

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
    np.random.seed(c.seed)
    
    #pygame.init()
    agentcbmgpu.initialize()
    total_goal = 0
    sum2_reward = 0
    mean_reward = 0#一定エピソードの報酬平均
    plus_reward_times = []#エピソード報酬を格納

    for episode in tqdm(range(c.num_episodes)):
        #環境の初期化
        agentcbmgpu.reset_network(episode,mean_reward)
        action = 2
        state, reward, done, info = env.reset()
        state, reward, done, info = env.step(action)
        sum1_reward = 0
        for t in range(c.max_number_of_steps):
            #action = get_action4()
            #action = agentcbmgpu.get_action(state, reward)
            #state, reward, done, info = env.step(action)
            #print(action)
            state, reward, done, info = env.step(action)
            action = agentcbmgpu.get_action(state, reward)

            sum1_reward += reward
            sum2_reward += reward
            if c.verbose:
                print("ep:%4d t:%4d done:%d r:%5.2f R:%5.2f a:%d q:%s s:%s" %
                (episode,agentcbmgpu.t,done,reward,sum1_reward,action,agentcbmgpu.q,agentcbmgpu.s))

            if c.render == 1:env.render()

            if t == c.max_number_of_steps-1:
                done = 1

            if done == 1:
                print("ep:%4d sum1_reward:%4f" % (episode,sum1_reward))
                plus_reward_times.append(sum1_reward)
                break

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    if plot == 1:
                        plot()
                    sys.exit()
                    
        #最新20エピソードの報酬総和の平均
        if episode>20:
            mean_reward = sum(plus_reward_times[episode-19:])/20

    env.close()
    plus_reward_times = np.array(plus_reward_times)
    plus_reward_times[plus_reward_times <0] = 0
    plus_reward_times[plus_reward_times >0] = 1
    c.plus_reward_times = plus_reward_times
    print("times of reward > 0",sum(plus_reward_times))
    plot(c.plot)

def plot(plot):
    if plot:
        fig=plt.figure(figsize=(8,6))#figsize=(8,6)
        ax = fig.add_subplot(5,1,1)
        ax.cla()
        ax.plot(agentcbmgpu.U)
        ax.set_ylabel("U")
        ax.set_ylim(-1,1)

        ax = fig.add_subplot(5,1,2)
        ax.cla()
        ax.plot(agentcbmgpu.Hx)
        ax.set_ylabel("Hx")
        ax.set_ylim(-1,1)

        ax = fig.add_subplot(5,1,3)
        ax.cla()
        ax.plot(agentcbmgpu.Hs)
        ax.set_ylabel("Hs")
        ax.set_ylim(-1,1)

        ax = fig.add_subplot(5,1,4)
        ax.cla()
        ax.plot(agentcbmgpu.R)
        ax.set_ylabel("R")
        ax.set_ylim(-1,1)

        ax = fig.add_subplot(5,1,5)
        ax.cla()
        ax.plot(agentcbmgpu.Q)
        ax.set_ylabel("Q")
        ax.set_xlabel("n")
        plt.show()
        plt.savefig("file_fig1")



if __name__=="__main__":
    if sys.argv:
        del sys.argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    env = environment.MyRobotEnv(c.render)
    agentcbmgpu = agentcbmgpu.Agent(config=c,plot = c.plot )
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)

    