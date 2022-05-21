# coding: utf-8
# Copyright (c) 2018 Katori Lab. All Rights Reserved
# NOTE: Grid cellをもした関数
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class HippocampalCell():
    num_grid=0
    num_hdcell=0
    def __init__(self):
        self.config_gridcell()
        self.config_hdcell()
        self.config_placecell()
    def gridcell(self,x,y,a,bx,by):
        return (np.cos(2*np.pi*(x/a-bx)) \
        + np.cos( 2*np.pi*( 0.5*(x/a-bx) + np.sqrt(3)/2.0*(y/a-by)) ) \
        + np.cos( 2*np.pi*(-0.5*(x/a-bx) + np.sqrt(3)/2.0*(y/a-by)) )+1.5)*2.0/9.0

    def hdcell(self,theta,theta0,sigma):
        d=np.arccos(np.cos(theta-theta0))/sigma
        return np.exp(-d*d)

    def config_gridcell(self):
        a0 = 40 # 基本周期
        na = 4 # 周期のパターン数
        nb = 3 # 空間の分割の数
        ng = na * nb * nb # number of grid cell グリッドセルの数
        self.cg = np.zeros(ng*3).reshape(ng,3)
        self.num_grid = ng
        m=0
        for i in range(nb):
            for j in range(nb):
                self.cg[m+0,:]=( 3*a0, i/nb, j/nb)
                self.cg[m+1,:]=( 5*a0, i/nb, j/nb)
                self.cg[m+2,:]=( 7*a0, i/nb, j/nb)
                self.cg[m+3,:]=(11*a0, i/nb, j/nb)
                m+=4
        return self.cg

    def config_placecell(self):#add
        x = 11
        y = 11
        self.interval = 90
        self.sigma = 90
        np = x * y
        self.num_place = np

        self.cp  = [[0 for i in range(2)] for j in range(np)]
        m = 0
        for i in range(x):
            for j in range(y):
                self.cp[m][0] = self.interval * i
                self.cp[m][1] = self.interval * j
                m+=1
        return self.cp

    def placecell(self,x,y,xi,yi):#add
        return np.exp(-(((x-xi)**2 + (y-yi)**2)/self.sigma**2))

    def encode_place(self,x,y):#add
        """
        空間上の座標の値からプレースセルの活動度を求めて返す。
        x,y: 空間上の座標
        vp：(戻り値)各グリットセルの活動度
        """
        vp = np.zeros(self.num_place)
        for i in range(self.num_place):
            vp[i] = self.placecell(x,y,self.cp[i][0],self.cp[i][1])
        return vp

    def decode_place(self,vp,x,y):
        sum=0
        for i in range(self.num_place):
            sum += self.placecell(x,y,self.cp[i][0],self.cp[i][1]) * vp[i]
        return sum

    def decode_place_map(self,xi,yi):
        x  = np.linspace(50,950,11)
        y = np.linspace(50,950,11)

        X, Y = np.meshgrid(x, y)
        Z = self.placecell(xi,yi,X,Y)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.pcolor(X,Y,Z)
        fig.colorbar(im)
        """
        r1 = patches.Rectangle(xy=(150, 150), width=300, height=700, ec='#000000', fill=True)
        r2 = patches.Rectangle(xy=(550, 150), width=300, height=700, ec='#000000', fill=True)
        ax.add_patch(r1)
        ax.add_patch(r2)
        """
        plt.axis([0, 1000, 1000, 0])
        plt.show()

    def decode_place_map2(self,vp):
        x = np.linspace(50,950,11)
        y = np.linspace(50,950,11)
        x,y = np.meshgrid(x,y)
        z = self.decode_place(vp,x,y)
        return x,y,z

    def encode_position(self,x,y):
        """
        空間上の座標の値からグリッドセルの活動度を求めて返す。
        x,y: 空間上の座標
        vg：(戻り値)各グリットセルの活動度
        """
        #num_grid = len(gc)
        vg = np.zeros(self.num_grid)
        for j in range(self.num_grid):
            vg[j] = self.gridcell(x,y,self.cg[j,0],self.cg[j,1],self.cg[j,2])
        return vg

    def decode_position(self,vg,x,y):
        """
        グリッドセルの活動度 vg を基に、位置x,yに対応することの尤もらしさを返す。
        """
        #self.num_grid = len(gc)
        sum=0
        for j in range(self.num_grid):
            sum += self.gridcell(x,y,self.cg[j,0],self.cg[j,1],self.cg[j,2]) * vg[j]
        return sum

    def decode_map(self,vg):
        #x = np.linspace(-1000,1000,101)
        #y = np.linspace(-1000,1000,101)
        x = np.linspace(50,950,101)
        y = np.linspace(50,950,101)
        x,y = np.meshgrid(x,y)
        z = self.decode_position(vg,x,y)
        return x,y,z

    def config_hdcell(self):
        self.num_hdcell=12
        self.sigma=0.5

    def encode_angle(self,theta):
        vh = np.zeros(self.num_hdcell)
        for i in range(self.num_hdcell):
            vh[i] = self.hdcell(theta,2*np.pi*i/self.num_hdcell,self.sigma)
        return vh

hc=HippocampalCell()

def test1():
    gc=hc.config_gridcell()
    print(gc)
    (xi,eta) = (100,150)
    vg=hc.encode_position(xi,eta)
    print(vg)

def test2():
    """
    グリッドセルの活動パターンを表示
    """
    ### Gird の表示
    x = np.linspace(-200,200,101)
    y = np.linspace(-200,200,101)
    x,y = np.meshgrid(x,y)

    z = hc.gridcell(x,y,100,1/2,1/2)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolor(x,y,z)
    fig.colorbar(im)
    plt.show()

def test3():
    """
    座標をエンコード、その後デコードしてマップ上に表示
    """
    (xi,eta) = (500,800)
    gc = hc.config_gridcell()
    vg = hc.encode_position(xi,eta)
    x,y,z = hc.decode_map(vg)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolor(x,y,z)
    fig.colorbar(im)
    """
    r1 = patches.Rectangle(xy=(150, 150), width=300, height=700, ec='#000000', fill=True)
    r2 = patches.Rectangle(xy=(550, 150), width=300, height=700, ec='#000000', fill=True)
    ax.add_patch(r1)
    ax.add_patch(r2)
    """
    plt.axis([0, 1000, 1000, 0])
    plt.show()

def test4():
    """
    ロボットが動きまわるときに、どのようなグリッドセルの活動の時間変動を確認する。
    """
    hc.config_gridcell()
    nt=500
    T = np.zeros(nt)
    X = np.zeros(nt)
    Y = np.zeros(nt)
    H = np.zeros(nt)
    GC = np.zeros(nt*hc.num_grid).reshape(nt,hc.num_grid)
    HC = np.zeros(nt*hc.num_hdcell).reshape(nt,hc.num_hdcell)

    for i in range(nt):
        t = 100*i/(nt-1)
        # rx,ry:ロボットの座標
        rx = 10+100*np.cos(t/10)
        ry = 100*np.sin(t/10)
        # ロボットの角度
        h = t/10

        T[i] = t
        X[i] = rx
        Y[i] = ry
        H[i] = h

        vg = hc.encode_position(rx,ry)
        vh = hc.encode_angle(h)

        GC[i,:]=vg
        HC[i,:]=vh

    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax.plot(T,X)
    ax.plot(T,Y)
    ax = fig.add_subplot(3,1,2)
    ax.plot(T,GC)
    ax = fig.add_subplot(3,1,3)
    ax.plot(T,HC)

    plt.show()

def test5():
    hc.config_hdcell()
    vh=hc.encode_angle(0)
    print(vh)

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
