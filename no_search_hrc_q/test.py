# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""

"""
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

### 共通設定
from main_HRC_agent_q import Config
config = Config()

common.config  = config
common.prefix  = "data%s_main_HRC_agent_q" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_main_HRC_agent_q" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python3 main_HRC_agent_q.py " # 実行されるプログラム
common.columns=[
    "id","plot",'seed',
    "train_episode_num","test_episode_num","max_step_num",
    "show_env","is_minus",
    'NN','Nu','Nh','Ny','Temp',
    'alpha_i','alpha_r','alpha_s','beta_i','beta_r',
    'ep_ini','ep_fin','ep_2',
    'eta_ini','eta_fin','eta_2',
    'gamma_wout','k_greedy',
    'test_prob','cnt_overflow','ave_reward']

common.parallel= 120
common.setup()
common.report_common()
common.report_config(config)


import warnings
warnings.simplefilter('ignore', FutureWarning)


### ランダムサーチ
def rs1():
    rs.clear()
    rs.append("alpha_r",min=0,max=5)
    rs.append("alpha_i",min=0,max=5)
    rs.random(num=60,samples=2)
    df = common.load_dataframe() # 直前に保存されたcsvファイルをデータフレーム(df)に読み込む
    df = df[['alpha_r','alpha_i','cnt_overflow','test_prob']] # 指定した列のみでデータフレームを構成する
    #df = df[(df['test_prob']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['test_prob'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）

    opt.append("beta_i",value=0.86,min=0.0,max=1,round=2)
    opt.append("alpha_i",value=0.4,min=0.0,max=1,round=2)

    opt.append("beta_r",value=0.58,min=0.0,max=1,round=2)
    opt.append("alpha_r",value=0.08,min=0.,max=1,round=2)

    opt.append("alpha_s",value=0.98,min=0,max=2,round=2)
    opt.append('ep_ini',value=0.01,min=0,max=0.1,round=5)

    opt.maximize(target='test_prob',iteration=30,population=40,samples=3)

    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
#optimize()

def plot1(x,y,ystd,ymin,ymax,color=None,width=1,label=None):
    # エラーバーをつけてグラフを描画、平均、標準偏差、最大値、最小値をプロットする。
    #ax.errorbar(x,y,yerr=ystd,fmt='o',color=color,capsize=2,label="xxxx")
    plt.plot(x,y,color=color,linestyle='-',linewidth=width,label=label)
    plt.fill_between(x,y-ystd,y+ystd,color=color,alpha=.2)
    plt.plot(x,ymin,color=color,linestyle=':',linewidth=1)
    plt.plot(x,ymax,color=color,linestyle=':',linewidth=1)

def gridsearch(X1,min=0,max=1,num=41,samples=10):
    # 指定された変数(X1)についてグリッドサーチを行い、評価基準の変化をまとめてプロット

    gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)
    df = common.load_dataframe()
    #print(df)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,8))

    plt.subplot(3,1,1)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"test_prob")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="test_prob")
    plt.ylabel("test_prob")
    plt.grid(linestyle="dotted")

    plt.subplot(3,1,2)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,'ave_reward')
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label='ave_reward')
    plt.ylabel('ave_reward')
    plt.grid(linestyle="dotted")

    plt.subplot(3,1,3)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"cnt_overflow")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(2),label="cnt_overflow")
    plt.ylabel("overflow")
    #plt.yscale('log')
    plt.grid(linestyle="dotted")
    #plt.ylim([0,1]) # y軸の範囲

    plt.xlabel(X1)
    vs.plt_output()

def gs2():
    ns=10
    #gridsearch("Nh",min=50,max=700,num=41,samples=ns)
    gridsearch("alpha_r",min=0.00,max=1,num=41,samples=ns)
    gridsearch("alpha_i",min=0.00,max=1,num=41,samples=ns)
    gridsearch("alpha_s",min=0.0,max=2,num=41,samples=ns)
    gridsearch("beta_r",min=0.00,max=1,num=41,samples=ns)
    gridsearch("beta_i",min=0.00,max=1,num=41,samples=ns)


    gridsearch("eta_2",min=0.00,max=0.1,num=101,samples=ns)
    gridsearch("ep_2",min=0.00,max=0.1,num=101,samples=ns)
    #gridsearch("Temp",min=-256,max=256,num=4001,samples=ns)

    #gridsearch("delay",min=5,max=100,num=41,samples=ns)
    #gridsearch("lambda0",min=0.01,max=1.5,num=41,samples=ns)
gs2()
