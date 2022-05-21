# Copyright (c) 2018 Katori lab. All Rights Reserved

import numpy as np
#import scipy.linalg

def generate_random_matrix(Nx,Ny,alpha,beta,distribution="one",normalization="sd",circle=False):
    """
    ランダムに行列を生成する。結合率（beta）の割合で非ゼロの値をランダムに割り当て、
    規格化の後、係数(alpha)をかけて出力する。
    Nx:行数
    Ny:列数
    alpha:スケールパラメータ
    beta: 結合率
    dist: ランダムに割り当てる値の分布 {one, normal, uniform}
        one: １または-１
        normal: 正規分布
        uniform: 一様分布
    normalization: 正規化{none,sr,sd}
        none: なにもしない。
        sr: スペクトル半径（最大固有値）で規格化
        sd: 列方向の和の分散が１になるように規格化


    2021/09/22
    超立方体上の疑似ビリヤードダイナミクスに基づくレザバー計算(HRC)の結合荷重生成プログラム
    変更なし
    """
    if circle:
        #taikaku = "zero"
        taikaku = "nonzero"
        Wr = np.zeros((Nx,Ny))
        for i in range(Nx-1):
            Wr[i,i+1] = 1
        Wr[-1,0]=1
        # #print(Wr)
        v = np.linalg.eigvals(Wr)
        lambda_max = max(abs(v))
        Wr = Wr/lambda_max*alpha
        return Wr

    W = np.zeros(Nx * Ny)
    nonzeros = int(Nx * Ny * beta)
    if distribution == "one":
        W[0:int(nonzeros / 2)] = 1
        W[int(nonzeros / 2):int(nonzeros)] = -1
        var = 1

    if distribution == "normal":
        W[0:nonzeros] = np.random.normal(0,1, nonzeros)
        var = 1

    if distribution == "uniform":
        W[0:nonzeros] = np.random.uniform(-1,1, nonzeros)
        var = 1/3

    np.random.shuffle(W)
    W = W.reshape((Nx, Ny))

    # spectral radium (sr) 最大固有値による規格化
    if normalization == "sr":
        assert Nx == Ny, "Nx and Ny should be same for spectral radius normalization"
        done = False
        if not done:
            v = np.linalg.eigvals(W)
            lambda_max = max(abs(v))
            if lambda_max>0:
                W = W / lambda_max
                done = True
            else:
                #NOTE 最大固有値が０の場合はシャッフルして再計算
                np.random.shuffle(W)

    # standard deviation (sd) 標準偏差による規格化
    # NOTE: 分散 var の分布から取得したm個のサンプルの和の分散は m * var となる。
    # 列方向の和の分散が１になるように規格化
    if normalization == "sd":
        W = W / np.sqrt(var * beta * Ny)

    W = W * alpha

    return W
