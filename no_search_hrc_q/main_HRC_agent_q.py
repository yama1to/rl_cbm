"""
2021/12/15
超立方体上の疑似ビリヤードダイナミクスに基づくレザバー計算(HRC)を用いたエージェント
グリッドサーチなし
Q-learningを実装
ε-greedy法を実装
"""
import argparse
import matplotlib.pyplot as plt
import copy
import time
from explorer import common

from collections import defaultdict
import gym
from agent import Agent
from frozen_lake_util import show_q_value
from model_HRC_q import HRC_q
import numpy as np

from tqdm import tqdm 



class Config:
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0
        self.seed:int=0 # 乱数生成のためのシード

        self.train_episode_num = 1000
        self.test_episode_num = 10

        self.show_env = 0 # 環境の表示 
        self.is_minus = 1 # 負の条件の有無
        self.report_interval_train = 50
        self.report_interval_test = 50
        self.max_step_num = 20# 1episodeの最大ステップ数

        self.NN = 256 # 1サイクル当たりの時間ステップ

        self.Nu = 16   # size of input
        self.Nh = 20 # size of dynamical reservior
        self.Ny = 4   # size of output
        
        self.Temp = 1.0
        self.dt = 1.0 / self.NN # 0.001
        
        self.alpha_i = 0.6#c.x1
        self.alpha_r = 1

        self.alpha_b = 0.
        self.alpha_s = 0.1

        self.beta_i = 0.3
        self.beta_r = 0.3

        self.ep_2 = 0.01
        self.ep_ini = 0.01
        self.ep_fin = 0.

        self.eta_2 = 0.01
        self.eta_ini = 0.1
        self.eta_fin = 0

        self.gamma_wout = 0.9 #0.915 # defaultは0.985最も良い0.925出力層の結合荷重の割引率γ
        self.k_greedy = 0.005 # ε-greedy法のεの決定係数

        self.test_prob = 0

        self.cnt_overflow = 0
        self.ave_reward = 0

class HRC_QlearnAgent(Agent, HRC_q):

    def __init__(self,c):
        Agent.__init__(self)
        HRC_q.__init__(self,c)
        self.rewards = []


    #負の報酬の条件追加:TODO
    def minus_reward(self, is_done, get_reward, step_count):
        if is_done and get_reward == 0: # 穴に落ちたらマイナスの報酬
            #print("fall into a hole")
            #get_reward = -0.5
            get_reward = -0.1
        elif get_reward != 1 and step_count >= c.max_step_num: # 最大ステップ数を超えたら終了
            #print("over step !!!")
            get_reward = -0.5
            is_done = True

        return is_done, get_reward


    def learn(self, env, train_epsd, test_epsd, render, minus):
        actions = list(range(env.action_space.n))# 可能な行動

        #Q値(dictionary型)表示用の配列--------------------
        #actionの数だけキーに対応する配列を用意する
        # action_space.n=4のとき、x:{0,0,0,0}となる
        self.output_train = defaultdict(lambda: [0] * len(actions))
        self.output_test = defaultdict(lambda: [0] * len(actions))
        #-------------------------------------------------
        
        # train
        total_steps=0
        goal_count_train_test = 0
        self.init_log_train()
        self.initialize_setting(actions, train_epsd, 0)
        for e in tqdm(range(train_epsd)):
            #print("現在の訓練episodeは{}".format(e))
            self.reset_network(e, goal_count_train_test)
            state = env.reset() # 環境のリセット(エージェントを初期の座標に配置)
            self.step_network_q(state)
            self.hrc_update_q()# hrcの出力の更新
            
            done = False #1episodeの終了判定
            n_step=0 #ステップ数の記録
            sum_reward = 0

            while not done:
                if render: env.render()

                action = self.policy_epsilon(self.rewards) # 最初の行動
                #OpenAIのライブラリ:TODO------------------------
                n_state, reward, done, info = env.step(action)
                #-----------------------------------------------
                if minus: #負の報酬の表示
                    done, reward = self.minus_reward(done, reward, n_step)
                else:
                    if reward != 1 and n_step >= c.max_step_num: done = True
                
                sum_reward += reward

                self.step_network_q(n_state)
                self.output_train[state] += self.yp_pre
                self.train_weight_q(action, reward,self.rewards)

                # update
                n_step += 1
                state = n_state

            else:
                self.log_train(reward)
                if reward == 1:
                    goal_count_train_test += 1
                    #print("-------------get reward------------")

            
            if e != 0 and e % c.report_interval_train == 0: 
                self.show_reward_log_train(c.report_interval_train, e)

            
            total_steps+= n_step
            self.rewards.append(sum_reward)
        # test

        goal_count_test = 0
        self.init_log_test()
        self.initialize_setting(actions, train_epsd+test_epsd, 1)



        for e in tqdm(range(test_epsd)):
            #print("現在のテストepisodeは{}".format(e))
            self.reset_network(e+train_epsd, goal_count_train_test)
            state = env.reset() # 環境のリセット(エージェントを初期の座標に配置)
            self.step_network_q(state)
            self.hrc_update_q()# hrcの出力の更新
            
            done = False #1episodeの終了判定
            n_step=0 #ステップ数の記録
            while not done:
                if render: env.render()

                action = self.policy_epsilon(self.rewards) # 最初の行動
                #OpenAIのライブラリ:TODO------------------------
                n_state, reward, done, info = env.step(action)
                #-----------------------------------------------
                if minus: #負の報酬の表示
                    done, reward = self.minus_reward(done, reward, n_step)
                else:
                    if reward != 1 and n_step >= c.max_step_num: done = True

                self.step_network_q(n_state)
                self.output_test[state] += self.yp_pre
                self.hrc_update_q()# hrcの出力の更新

                # update
                n_step += 1
                state = n_state

            else:
                self.log_test(reward)
                if reward == 1:
                    goal_count_test += 1
                    goal_count_train_test += 1
                    #print("-------------get reward------------")

            if e != 0 and e % c.report_interval_test == 0: 
                self.show_reward_log_test(c.report_interval_test, e)
            total_steps+= n_step

        test_prob = goal_count_test/test_epsd
        print("test episodeの正答率{}!!".format(test_prob))
        c.test_prob = test_prob

        self.cnt_overflow = self.cnt_overflow/self.Nh/(c.train_episode_num+c.test_episode_num)
        c.cnt_overflow = self.cnt_overflow
        print(self.cnt_overflow)
        
        c.ave_reward = np.mean(self.rewards)


    
        plt.figure(figsize=(6,8))
        

        plt.subplot(2,1,1)
        plt.stackplot(list(range(len(self.rewards))),self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        #plt.ylim(-1,1)
        plt.title("Nx=%d, alpha_i=%.2lf, alpha_r=%.2lf, alpha_b=%.2lf,\n alpha_s=%.2lf, beta_i=%.2lf, beta_r=%.2lf,\n ep_2=%.4lf, ep_ini=%.4lf, ep_fin=%.4lf, eta_2=%.4lf,\n eta_ini=%.4lf, eta_fin=%.4lf, gamma_wout=%.3lf" %
            (c.Nh, c.alpha_i, c.alpha_r,c.alpha_b,
            c.alpha_s, c.beta_i,c.beta_r,
            c.ep_2,c.ep_ini,c.ep_fin,
            c.eta_2,c.eta_ini,c.eta_fin,c.gamma_wout)
            )
        #plt.savefig('./figs/'+common.string_now()+"reward")

        plt.subplot(2,1,2)
        plt.plot(self.Eta_collect)
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        #plt.ylim(0,1)
        
        plt.grid(linestyle="dotted")
        # plt.savefig('./figs/'+common.string_now()+"learning_rate")
        
        plt.savefig('./figs/'+common.string_now()+str(c.seed)+"")



def train(c):
    c.seed = int(c.seed)
    np.random.seed(c.seed)
    agent = HRC_QlearnAgent(c)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, c.train_episode_num, c.test_episode_num, c.show_env, c.is_minus)
    
    if c.plot:
        # agent.plot_hrc_episode() # hrcの内部状態を表示(episodeごと)
        # agent.plot_hrc_step()# hrcの内部状態を表示(stepごと)
        # agent.plot_hrc_out() # hrcの出力のみ表示
        show_q_value(agent.output_train) # Q値を表示
        show_q_value(agent.output_test) # Q値を表示
        agent.show_reward_log_train() # 報酬の履歴を表示
        agent.show_reward_log_test() # 報酬の履歴を表示



if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    
    train(c)
    if a.config: common.save_config(c)
