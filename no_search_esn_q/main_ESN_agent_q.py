"""
2021/12/15
エコーステートネットワーク(ESN)を用いたエージェント
グリッドサーチなし
Q-learningを実装
ε-greedy法を実装
"""

from collections import defaultdict
import gym
from agent import Agent
from frozen_lake_util import show_q_value
from model_ESN_q import ESN_q
import numpy as np

random_seed = 0
train_episode_num = 1000
test_episode_num = 100
show_env = False # 環境の表示
is_minus = 1 # 負の条件の有無
report_interval_train = 50
report_interval_test = 50
max_step_num = 20# 1episodeの最大ステップ数

rewards = []

class ESN_QlearnAgent(Agent, ESN_q):

    def __init__(self):
        Agent.__init__(self)
        ESN_q.__init__(self)


    #負の報酬の条件追加:TODO
    def minus_reward(self, is_done, get_reward, step_count):
        if is_done and get_reward == 0: # 穴に落ちたらマイナスの報酬
            print("fall into a hole")
            #get_reward = -0.5
            get_reward = -0.1
        elif get_reward != 1 and step_count >= max_step_num: # 最大ステップ数を超えたら終了
            print("over step !!!")
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
        goal_count_train_test = 0
        self.init_log_train()
        self.initialize_setting(actions, train_epsd, 0)
        for e in range(train_epsd):
            print("現在の訓練episodeは{}".format(e))
            self.reset_network(e, goal_count_train_test)
            state = env.reset() # 環境のリセット(エージェントを初期の座標に配置)
            self.step_network_q(state)
            self.esn_update_q()# esnの出力の更新
            
            done = False #1episodeの終了判定
            n_step=0 #ステップ数の記録
            while not done:
                if render: env.render()

                action = self.policy_epsilon(rewards) # 最初の行動
                #OpenAIのライブラリ:TODO------------------------
                n_state, reward, done, info = env.step(action)
                #-----------------------------------------------
                if minus: #負の報酬の表示
                    done, reward = self.minus_reward(done, reward, n_step)
                else:
                    if reward != 1 and n_step >= max_step_num: done = True
                rewards.append(reward)
                self.step_network_q(n_state)
                self.output_train[state] += self.q
                self.train_weight_q(action, reward,rewards)

                # update
                n_step += 1
                state = n_state

            else:
                self.log_train(reward)
                if reward == 1:
                    goal_count_train_test += 1
                    print("-------------get reward------------")

            if e != 0 and e % report_interval_train == 0: 
                self.show_reward_log_train(report_interval_train, e)

        # test
        goal_count_test = 0
        self.init_log_test()
        self.initialize_setting(actions, train_epsd+test_epsd, 1)
        for e in range(test_epsd):
            print("現在のテストepisodeは{}".format(e))
            self.reset_network(e+train_epsd, goal_count_train_test)
            state = env.reset() # 環境のリセット(エージェントを初期の座標に配置)
            self.step_network_q(state)
            self.esn_update_q()# esnの出力の更新
            
            done = False #1episodeの終了判定
            n_step=0 #ステップ数の記録
            while not done:
                if render: env.render()

                action = self.policy_epsilon(rewards) # 最初の行動
                #OpenAIのライブラリ:TODO------------------------
                n_state, reward, done, info = env.step(action)
                #-----------------------------------------------
                if minus: #負の報酬の表示
                    done, reward = self.minus_reward(done, reward, n_step)
                else:
                    if reward != 1 and n_step >= max_step_num: done = True
                rewards.append(reward)
                self.step_network_q(n_state)
                self.output_test[state] += self.q
                self.esn_update_q()# esnの出力の更新

                # update
                n_step += 1
                state = n_state

            else:
                self.log_test(reward)
                if reward == 1:
                    goal_count_test += 1
                    goal_count_train_test += 1
                    print("-------------get reward------------")

            if e != 0 and e % report_interval_test == 0: 
                self.show_reward_log_test(report_interval_test, e)

        test_prob = goal_count_test/test_epsd
        print("test episodeの正答率{}!!".format(test_prob))
        print(goal_count_train_test/(train_episode_num+test_episode_num))


def train():
    np.random.seed(random_seed)
    agent = ESN_QlearnAgent()
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, train_episode_num, test_episode_num, show_env, is_minus) 
    agent.plot_esn() # esnの内部状態を表示
    agent.plot_esn_out() # esnの出力のみ表示
    show_q_value(agent.output_train) # Q値を表示
    show_q_value(agent.output_test) # Q値を表示
    agent.show_reward_log_train() # 報酬の履歴を表示
    agent.show_reward_log_test() # 報酬の履歴を表示


if __name__ == "__main__":
    train()
