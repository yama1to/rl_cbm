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
from maincbm import Config
config = Config()
common.config  = config
common.prefix  = "data%s_maincbm" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_maincbm" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python3 maincbm.py " # 実行されるプログラム
common.columns=[
    "id","plot",'seed',"num_episodes","max_steps","render",
    'alpha_i','alpha_r','alpha_s','beta_i','beta_r',
    'gamma','tau_x','eta_2','eta_ini','eta_fin','ep_2','ep_ini','ep_fin',
    'cnt_overflow','success1','success2','ave_rew'
                ]

common.parallel= 32
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
    df = df[['alpha_r','alpha_i','cnt_overflow','success1']] # 指定した列のみでデータフレームを構成する
    #df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    #opt.append("Nh",value=500,min=300,max=1000,round=1)
    

    opt.append("beta_i",value=0.1,min=0.0,max=1,round=2)
    opt.append("alpha_i",value=0.6,min=0.6,max=0.6,round=0)

    opt.append("beta_r",value=0.1,min=0.0,max=1,round=2)
    opt.append("alpha_r",value=1,min=1.,max=1,round=0)

    opt.append("alpha_s",value=0.5,min=0,max=2,round=2)

    # opt.append("gamma",value=1,min=0,max=2,round=2)
    # opt.append("tau_x",value=1,min=0,max=2,round=2)

    # opt.append("eta_2",value=0.18,min=0,max=2,round=3)
    opt.append("eta_ini",value=0.001,min=0,max=0.1,round=6)
    # opt.append("eta_fin",value=0.,min=0,max=2,round=6)

    # opt.append("ep_2",value=0.17,min=0,max=1,round=3)
    # opt.append("ep_ini",value=0.62,min=0,max=1,round=3)
    # opt.append("ep_fin",value=0.01,min=0,max=1,round=3)

    # opt.append("beta_b",value=0.1,min=0.0,max=1,round=2)
    # opt.append("alpha_b",value=1,min=0.0,max=1,round=2)
    #opt.append("alpha0",value=1,min=0,max=1,round=2)
    #opt.append("Temp",value=10,min=1,max=10,round=2)
    # opt.maximize(target="success1",iteration=30,population=40,samples=3)
    #opt.maximize(target="success1",iteration=30,population=40,samples=3)
    opt.maximize(target="success1",iteration=20,population=30,samples=3)
    #opt.minimize(target="success1",iteration=10,population=3,samples=3)
    #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)
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

    plt.subplot(1,1,1)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"success1")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="success1")
    plt.ylabel("success1")
    plt.grid(linestyle="dotted")

    plt.subplot(3,1,2)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"ave_rew")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="ave_rew")
    plt.ylabel("ave_rew")
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
    gridsearch("alpha_r",min=0.,max=1,num=41,samples=ns)
    gridsearch("alpha_i",min=0.0,max=1,num=41,samples=ns)
    gridsearch("alpha_s",min=0.0,max=2,num=41,samples=ns)
    # #gridsearch("alpha_b",min=0.0,max=2,num=41,samples=ns)
    # # gridsearch("alpha_o",min=0.0,max=2,num=41,samples=ns)

    # # gridsearch("beta_o",min=0.0,max=1,num=41,samples=ns)
    # #gridsearch("beta_b",min=0.0,max=1,num=41,samples=ns)
    gridsearch("beta_i",min=0.0,max=1,num=41,samples=ns)
    gridsearch("beta_r",min=0.0,max=1,num=41,samples=ns)
    # # gridsearch("Temp",min=1,max=10,num=41,samples=ns)
    # # gridsearch("ave",min=-1,max=1,num=41,samples=ns)
    # # gridsearch("std",min=0.01,max=1,num=41,samples=ns)



    gridsearch('gamma',min=0.0,max=1,num=41,samples=ns)
    # gridsearch('tau_x',min=0.0,max=1,num=41,samples=ns)
    gridsearch('eta_2',min=0.0,max=1,num=41,samples=ns)
    gridsearch('eta_ini',min=0.0,max=1,num=41,samples=ns)
    gridsearch('eta_fin',min=0.0,max=1,num=41,samples=ns)
    gridsearch('ep_2',min=0.0,max=1,num=41,samples=ns)
    gridsearch('ep_ini',min=0.0,max=1,num=41,samples=ns)
    gridsearch('ep_fin',min=0.0,max=1,num=41,samples=ns)

gs2()
