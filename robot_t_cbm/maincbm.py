"""
よく変更するもの
# c.num_episodes エピソードの回数
# c.max_steps エピソードあたりの最大ステップ数
# c.test_episode テストに使用するエピソード数
# eva_samples  使用するサンプル数。異なるシードで状態をリセットし、動作する。
# c.render シュミレーションを表示。0で非表示、1で表示。
"""

from unittest.util import _count_diff_hashable
import environment
import main_agentcbm
import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
import pandas as pd
from explorer import common
import pickle
import argparse
from explorer import common
import someplot
from tqdm import tqdm 



### パラメータ探索用 ###
class Config():
    def __init__(self):
        ## columns, csv, id: データの管理のために必須の変数 ##
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0# 図の出力のオンオフ

        self.seed= 0 #乱数生成のためのシード
        self.num_episodes = 500#エピソードの回数(10000くらいでできる)
        self.max_steps = 300#最大ステップ
        self.test_episodes = 100#テストエピソード数


        self.render = 0#シミュレーション表示

        self.now_epi = 0
        ## config ##
        
        
        self.NN = 2**8


        self.Nu = 8 #size of input
        self.Nu2 = 2 #size of input signal
        self.Nx = 200#size of dynamical reservior
        self.Ny = 3 #size of output

        ### Hyperparameter ###
        # weight #
        self.alpha_r = 0.34 #Spectral radius
        self.beta_r = 0.8 #Bond density
        self.alpha_i = 0.4 #Input weight
        self.beta_i = 0.2  #Bond density
        self.gamma = 0.95 #減衰率　
        self.tau_x = 6 #時定数

        self.alpha_s = 0
        self.Temp = 1

        # 学習率(Step size parameter) #
        self.eta_2 = 0.18
        self.eta_ini = 0.003
        self.eta_fin = 0.00
        
        # 探索(epsilon) #
        self.ep_2 = 0.17
        self.ep_ini = 0.62
        self.ep_fin = 0.

        # other #
        self.alpha_x = 1.0
        self.alpha0 = 1.0
        self.beta = 1.0
        #self.sigma_np = -5
        self.cnt_overflow = 0

        self.rewards = []
        self.Eta_collect = []

        self.success1 = 0
        self.success2 = 0
        self.sum_rew = 0
        self.ave_rew=0

### 実行 ###
def execute(c):
    ## 初期化 ##
    np.random.seed(int(c.seed))
    env = environment.MyRobotEnv(c.render,c)
    agent = main_agentcbm.Agent(c)
    agent.initialize()#初期値設定
    ## 評価用変数 ##
    X, Y = [],[]#全エピソードの軌跡を格納

    c.num_episodes = int(c.num_episodes)
    c.max_steps = int(c.max_steps)
    success = [0] * c.num_episodes
    success2 = [0] * c.num_episodes
    fail = [0] * c.num_episodes
    conflict = [0] * c.num_episodes

    count= 0

    ### エピソード ###
    for episode in tqdm(range(c.num_episodes)):
        ## 環境の初期化 ##
        c.sum_rew = 0
        agent.reset_network(episode,success)#学習系を初期化
        state, reward, done, info = env.reset()#環境情報などを初期化
        c.sum_rew += reward
        
        action = agent.a#初期行動はランダム
        
        prev_r = 0
        sreward = 0
        ### ステップ ###
        for t in range(c.max_steps):
            ## 更新 ##
            state, reward, done, info = env.step(action)#actionから環境を更新
            c.sum_rew += reward
            sreward +=reward
            action = agent.get_action(state,reward,env.requirement)#学習と行動コマンド生成
            

            count += 1
            ## 終了条件 ##
            if done == 1 or t == c.max_steps-1:
                break

            ## ロボットのシュミレーションを表示 ##
            if c.render == 1: 
                env.render()
                ## パイゲームの終了 ##
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                        
        

            
        if c.plot & episode >= c.num_episodes-20:
            #ロボットの軌道を保存する
            plt.plot(c.sum_rew)

        if 2 <= episode < c.num_episodes-1:
            tmp = np.sum( np.heaviside( np.fabs(agent.r-prev_r) - 0.6 ,0))
            c.cnt_overflow += tmp
        prev_r = agent.r

        ### 評価&プロット ###
        ## ロボットの軌道を保存する(全てのエピソードを保存) ##
        X.append(env.x_oribit)
        Y.append(env.y_oribit) 
        c.rewards.append(sreward)
        c.now_epi +=1
        ## プロット(選択したエピソード) ##
        if c.plot & episode >= c.num_episodes-20:
            #ロボットの軌道を保存する
            X.append(env.x_oribit)
            Y.append(env.y_oribit)
            #someplot.plot_internal(episode,agent.Q,agent.U,agent.R)
            # someplot.plot_orbit(X,Y,episode-(c.num_episodes-20))#エピソードごとの軌道をプロットする
            # someplot.plot_internal(episode,agent.X,agent.Q,agent.U,agent.R)#内部状態をプロットする
            # plt.savefig("./figs/"+common.string_now()+'sum_reward')
            # plt.cla()
        ## 評価(1エピソードごとの結果を格納) ##
        success[episode] = env.check_eva[0]
        fail[episode] = env.check_eva[1]
        conflict[episode] = env.check_eva[2]
        success2[episode] = env.check_eva[3]

    if c.plot :
        plt.plot(agent.Eta)
        plt.show()
        plt.plot(agent.R)
        plt.show()
        plt.plot(agent.Hs)
        plt.plot(agent.Hx)
        plt.show()
    ## 評価(複数シードごとに) ##
    # 複数の評価エピソードの成功率を入れる
    # success_sum.append(sum(success[c.num_episodes-c.test_episode:])/c.test_episode)
    # fail_sum.append(sum(fail[c.num_episodes-c.test_episode:])/c.test_episode)
    # #conflict_sum.append(sum(conflict[c.num_episodes-c.test_episode:])/c.test_episode)
    # success2_sum.append(sum(success2[c.num_episodes-c.test_episode:])/c.test_episode)

    ## プロット(全エピソードの) ##
    someplot.plot_orbit_all(X,Y,c.seed,c.num_episodes)#全ての軌道を重ねてプロット
    
    #c.success1,c.success2 = someplot.print_evaluation(success,success2,c.num_episodes)#成功率をターミナルに表示
    print(c.success1,c.success2)
    c.ave_rew = np.mean(c.rewards)

    c.cnt_overflow = c.cnt_overflow/c.Nx/count
    print('cnt_overflow',c.cnt_overflow)

    # plt.figure(figsize=(6,8))

    # plt.subplot(2,1,1)
    # plt.stackplot(list(range(len(c.rewards))),c.rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # #plt.savefig('./figs/'+common.string_now()+"reward")

    # plt.subplot(2,1,2)
    # plt.plot(agent.Eta)
    # plt.xlabel('Episode')
    # plt.ylabel('Learning Rate')
    
    # plt.grid(linestyle="dotted")
    # # plt.savefig('./figs/'+common.string_now()+"learning_rate")
    # plt.savefig('./figs/'+common.string_now()+"")
    
    plt.figure(figsize=(6,8))
        

    plt.subplot(2,1,1)
    plt.stackplot(list(range(len(c.rewards))),c.rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(linestyle="dotted")
    #plt.ylim(-1,1)
    plt.title("Nx=%d, alpha_i=%.2lf, alpha_r=%.2lf,\n alpha_s=%.2lf, beta_i=%.2lf, beta_r=%.2lf,\n ep_2=%.4lf, ep_ini=%.4lf, ep_fin=%.4lf, eta_2=%.4lf,\n eta_ini=%.4lf, eta_fin=%.4lf, gamma_wout=%.3lf" %
        (c.Nx, c.alpha_i, c.alpha_r,#c.alpha_b,
        c.alpha_s, c.beta_i,c.beta_r,
        c.ep_2,c.ep_ini,c.ep_fin,
        c.eta_2,c.eta_ini,c.eta_fin,c.gamma)
        )
    #plt.savefig('./figs/'+common.string_now()+"reward")

    plt.subplot(2,1,2)
    plt.plot(agent.Eta)
    plt.xlabel('Episode')
    plt.ylabel('Learning Rate')
    #plt.ylim(0,1)
    
    plt.grid(linestyle="dotted")
    # plt.savefig('./figs/'+common.string_now()+"learning_rate")
    
    plt.savefig('./figs/'+common.string_now()+"")

    #someplot.plot_hist(success,c.num_episodes,seed_e)#ヒストグラムを作成
    # pca(下3行) #
    #X_pca = np.concatenate([agent.X_pca_right, agent.X_pca_left])#右と左で組み合わせた
    #x_0,x_1,x_2 = someplot.pre_pca(X_pca)
    #someplot.plot_pca(seed_e,x_0,x_1,x_2,agent.X_pca_right)#全部のエピソード合体

    ### 探索用 ###
    #c.y3 = sum(success[c.num_episodes-c.test_episode:])/c.test_episode
    
    env.close()


# ### メイン　###
# if __name__ == "__main__":
#     someplot.mkdir()
#     eva_samples = 1#サンプル数
#     success_sum, success2_sum, fail_sum, conflict_sum = [],[],[],[]
#     ## メイン、サンプル数を実行 ##
#     for i in range(eva_samples):
#         c = Config()
#         execute(c, i)

#     ## プロット(全シードの) ##
#     someplot.plot_hakohige(success_sum,success2_sum,fail_sum)#箱ヒゲ図
#     #someplot.plot_csv(csv_list)
    
        
#'''
### パラメーター探索用 ###
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config() # デフォルト設定 グローバル変数
    # プログラム実行の際に -configが指定された場合には、設定を読み込み実行する。
    if a.config: c=common.load_config(a) # config引数による設定の読み込み。
    execute(c)# cに書き込まれた設定に基づいて実行、cに実行結果が書き込まれる。
    if a.config: common.save_config(c) # 実行結果を保存する。
    #if c.plot: plot(c) #エラーがでるからけしてしまった¥
   # '''
    
