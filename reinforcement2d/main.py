# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)

import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import sys
import agent
import environment

seed = 0
num_episodes = 500#エピソードの回数
max_number_of_steps = 200#最大ステップ
render = 0 #シミュレーション表示
plot = 1#plot表示
verbose = 0

env = environment.MyRobotEnv(render)
agent = agent.Agent()

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

def execute():
    np.random.seed(seed)
    #pygame.init()
    agent.initialize()
    total_goal = 0
    sum2_reward = 0
    mean_reward = 0#一定エピソードの報酬平均
    sum1_reward_list = []#エピソード報酬を格納

    for episode in range(num_episodes):
        #環境の初期化
        agent.reset_network(episode,mean_reward)
        action = 2
        state, reward, done, info = env.reset()
        state, reward, done, info = env.step(action)
        sum1_reward = 0
        for t in range(max_number_of_steps):
            #action = get_action4()
            #action = agent.get_action(state, reward)
            #state, reward, done, info = env.step(action)
            state, reward, done, info = env.step(action)
            action = agent.get_action(state, reward)

            sum1_reward += reward
            sum2_reward += reward
            total_goal += env.goal
            if verbose:
                print("ep:%4d t:%4d done:%d r:%5.2f R:%5.2f a:%d q:%s s:%s" %
                (episode,agent.t,done,reward,sum1_reward,action,agent.q,agent.s))

            if render == 1:env.render()

            if t == max_number_of_steps-1:
                done = 1

            if done == 1:
                print("ep:%4d sum1_reward:%4f" % (episode,sum1_reward))
                sum1_reward_list.append(sum1_reward)
                break

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    if plot == 1:
                        plot()
                    sys.exit()
        print(t)   
        #最新20エピソードの報酬総和の平均
        if episode>20:
            mean_reward = sum(sum1_reward_list[episode-19:])/20

    env.close()
    # print(mean_reward)
    # print(sum2_reward/max_number_of_steps)
    # sum1_reward_times = np.array(sum1_reward_lists)
    # sum1_reward_times = int(np.sum(np.heaviside(sum1_reward_times,0)))
    # print(sum1_reward_times)
    print(total_goal)


def plot():

    fig=plt.figure(figsize=(8,6))#figsize=(8,6)
    ax = fig.add_subplot(4,1,1)
    ax.cla()
    ax.plot(agent.U)
    ax.set_ylabel("U")
    ax.set_ylim(-1,1)

    ax = fig.add_subplot(4,1,2)
    ax.cla()
    ax.plot(agent.X)
    ax.set_ylabel("X")
    ax.set_ylim(-1,1)

    ax = fig.add_subplot(4,1,3)
    ax.cla()
    ax.plot(agent.R)
    ax.set_ylabel("R")
    ax.set_ylim(-1,1)

    ax = fig.add_subplot(4,1,4)
    ax.cla()
    ax.plot(agent.Q)
    ax.set_ylabel("Q")
    ax.set_xlabel("n")
    plt.show()
    plt.savefig(file_fig1)

if __name__=="__main__":
    execute()
