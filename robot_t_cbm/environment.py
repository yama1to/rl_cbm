# coding: utf-8
#８の字迷路（600mm,350mm）
import math
import numpy as np
import pygame
from pygame.locals import *
import sys

def get_intersection(rx,ry,sx,sy,x1,y1,x2,y2):
    """
    交点を取得する
    ①ロボットの中心とセンサの座標の直線
    ②壁や障害物の直線
    ①と②の交点を取得し、交点を返す
    (rx,ry):ロボの中心座標
    (sx,sy):伸ばしているセンサの向きの座標
    (x1,y1)(x2,y2):壁の座標
    (cx,cy):壁とセンサの交点
    """
    fx,fy=10000,10000

    a0 = sx - rx
    b0 = sy - ry
    a2 = x2 - x1
    b2 = y2 - y1

    d = a0*b2 - a2*b0
    if d == 0:#２直線が並行のとき（or重なっているとき)
        return False,fx,fy

    sn = b2 * (x1-rx) - a2 * (y1-ry)#計算
    s = sn/d#①の媒介変数
    #cx = rx + a0*sn/d#交点
    #cy = ry + b0*sn/d

    tn = b0 * (x1-rx) - a0 * (y1-ry)#なんか計算
    t = tn/d#②の媒介変数
    cx = x1 + a2*tn/d#交点
    cy = y1 + b2*tn/d

    if 0 <= t <= 1:#交点が線分（壁）の中にあるか
        if 0<s:#ベクトルがセンサ座標の方向に伸びているとき。sは媒介変数
            return True,cx,cy

    return False,fx,fy

def distance(a,b,c,d):#距離を測る
    x = np.array([a,b])
    y = np.array([c,d])
    u = y-x
    dis = np.linalg.norm(u)
    return dis

def get_nearest_intersection( rx, ry, sx,sy ,walls):
    """
    各距離センサーについて、すべての壁との距離を測り、最も近くの壁との距離を返す。
    rx,ry: ロボット位置の座標
    theta: センサーの向き
    walls: 壁の集合
    """
    cx = rx#エラーの原因
    cy = ry
    cd = 10000
    cross = False
    for j in range(len(walls)):# すべての壁について
        wall = walls[j]
        for i in range(len(wall)):
            if i < len(wall)-1:
                wx0,wy0 = wall[i  ][0],wall[i  ][1]
                wx1,wy1 = wall[i+1][0],wall[i+1][1]
            else:
                wx0,wy0 = wall[i][0],wall[i][1]
                wx1,wy1 = wall[0][0],wall[0][1]
            cross,bx,by = get_intersection(rx,ry,sx,sy,wx0,wy0,wx1,wy1)
            if cross:
                if distance(rx,ry,bx,by)<cd:
                    cx=bx
                    cy=by
                    cd=distance(rx,ry,cx,cy)
    return cd,cx,cy

def get_distance_to_wall( rx, ry, theta, degree, walls):
    """
    距離センサーで壁までの距離を測る。
    rx,ry,theta:ロボットの座標と向き
    degree: センサーの向き
    walls: 壁の集合
    poi_d: 交点までの距離
    poi_p: 交点の座標
    """
    poi_d = [] # points of intersection (distance)
    poi_p = [] # points of intersection (position)
    for i in range(len(degree)):# すべての距離センサーについて
        angle = theta + np.deg2rad(degree[i])
        sx = rx + np.cos(angle) # 伸ばしているセンサの向きの座標
        sy = ry - np.sin(angle)
        pp=[0,0]
        pd,pp[0],pp[1] = get_nearest_intersection(rx,ry,sx,sy,walls)
        poi_d.append(pd)
        poi_p.append(pp)
    return poi_d,poi_p

def intersect(p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y):
    """
    p1(p1x,p1y)とp2(p2x,p2y)を結ぶ線分と、p3(p3x,p3y)とp4(p4x,p4y)を結ぶ線分が交差するときTrueを返す。
    """
    tc1 = (p1x - p2x) * (p3y - p1y) + (p1y - p2y) * (p1x - p3x)
    tc2 = (p1x - p2x) * (p4y - p1y) + (p1y - p2y) * (p1x - p4x)
    td1 = (p3x - p4x) * (p1y - p3y) + (p3y - p4y) * (p3x - p1x)
    td2 = (p3x - p4x) * (p2y - p3y) + (p3y - p4y) * (p3x - p2x)
    return tc1*tc2<0 and td1*td2<0

def check_intersection(px0,py0,px1,py1,objects):
    """
    オブジェクト（壁など）との交差をチェックする。
    (px0,py0): 前の時間ステップにおけるロボット位置
    (px1,py1): 現在の時間ステップにおけるロボット位置
    objects: オブジェクトの集合
    cross: 交差した場合は True を返す。交差したオブジェクトのインデックスも返す。
    """
    cross = False
    i_cross = -1
    for j in range(len(objects)):
        object = objects[j]
        for i in range(len(object)):
            if i < len(object)-1:
                wx0,wy0 = object[i  ][0],object[i  ][1]
                wx1,wy1 = object[i+1][0],object[i+1][1]
            else:
                wx0,wy0 = object[i][0],object[i][1]
                wx1,wy1 = object[0][0],object[0][1]

            #print(i,len(object),px0,py0,px1,py1,w0x,w0y,w1x,w1y)
            if intersect(px0,py0,px1,py1,wx0,wy0,wx1,wy1):
                cross = True
                i_cross = j
                return cross, i_cross
                #px = px0
                #py = py0
    return cross, i_cross
    #return px,py,cross

def calcurate_wheel(action):
    """
    離散値で与えられる行動命令をロボットの動作速度に変換する。
    action: 0,1,2のいずれかの値をとる行動命令
    v:ロボットの速度
    omega:ロボットの回転角速度
    """
    d = 25#ロボットの中心からwheelまでの距離
    wheel_radius = 5#車輪の半径

    if action == 0:
        phiR = 8#回転角度Right
        phiL = 4#回転角度Left
    if action == 1:#Right
        phiR = 4#回転角度Right
        phiL = 8#回転角度Left
    if action == 2:
        phiR = 8#回転角度Right
        phiL = 8#回転角度Left
    #if action == 3:
    #    phiR = -8#回転角度Right
    #    phiL = -8#回転角度Left

    vR = wheel_radius * phiR#車輪の速度Right
    vL = wheel_radius * phiL#車輪の速度Left
    omega = (vR - vL)/ (2 * d) # ロボットの回転角速度
    v = (vR + vL)/ 2#ロボットの中心速度
    return v,omega

def detect_pass(px0,py0,px1,py1,line):
    """
    報酬線や基準線に対して通過と通過方向を判定
    (px0,py0): 前の時間ステップにおけるロボット位置
    (px1,py1): 現在の時間ステップにおけるロボット位置
    line: 2つの座標からなる線[[x1,y1],[x2,y2]]
    """
    detect = [False,'none','none']
    if intersect(px0,py0,px1,py1,line[0][0],line[0][1],line[1][0],line[1][1]):#交差
        detect[0] = True
        if px1-px0 > 0:
            detect[1] = 'toright'
        if px1-px0 < 0:
            detect[1] = 'toleft'
        if py1-py0 > 0:
            detect[2] = 'todown'
        if py1-py0 < 0:
            detect[2] = 'toup'

    return detect

### ロボットの四隅で衝突判定を行う場合 ###
def check_intersection2(robot_corners,objects):
    """
    オブジェクト（壁など）との交差をチェックする。
    robot_corners: ロボットの四隅の座標
    objects: オブジェクトの集合
    cross: 交差した場合は True を返す。交差したオブジェクトのインデックスも返す。
    """
    cross = False
    i_cross = -1
    for k in range(len(robot_corners)):
        if k < len(robot_corners)-1:
            x0,y0 = robot_corners[k  ][0],robot_corners[k  ][1]
            x1,y1 = robot_corners[k+1][0],robot_corners[k+1][1]
        else:
            x0,y0 = robot_corners[k][0],robot_corners[k][1]
            x1,y1 = robot_corners[0][0],robot_corners[0][1]
        for j in range(len(objects)):
            object = objects[j]
            for i in range(len(object)):
                if i < len(object)-1:
                    wx0,wy0 = object[i  ][0],object[i  ][1]
                    wx1,wy1 = object[i+1][0],object[i+1][1]
                else:
                    wx0,wy0 = object[i][0],object[i][1]
                    wx1,wy1 = object[0][0],object[0][1]
                #print(i,len(object),px0,py0,px1,py1,w0x,w0y,w1x,w1y)
                if intersect(x0,y0,x1,y1,wx0,wy0,wx1,wy1):
                    cross = True
                    i_cross = j
                    return cross, i_cross
                    #px = px0
                    #py = py0
    return cross, i_cross
    #return px,py,cross

### ロボットの四隅で衝突判定を行う場合 ###
def calluculate_robot(rx,ry,l,theta):
    """
    ロボットの四隅の座標を取得する関数
    rx,ry ロボットの座標
    l ロボットの中心から車輪まで
    theta ロボットの回転角
    """
    robot_new = []
    robot = [[-l,-l],[l,-l],[l,l],[-l,l]]#ロボットの中心に対して
    #robot = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]#ロボットの四隅座標
    for i in range(len(robot)):#theta分回転させる
        point = robot[i]
        x1 = (point[0]*math.cos(theta) - point[1]*math.sin(theta))+rx
        x2 = (point[0]*math.sin(theta) + point[1]*math.cos(theta))+ry
        robot_new.append([x1,x2])

    return robot_new
    


    
### 環境メイン ###
class MyRobotEnv:
    def __init__(self,render,c):
        ## 初期設定 ##
        self.viewer = 0
        self.reward = 0#タイムステップごとの報酬
        self.t = 0#タイムステップ
        self.state = None
        self.requirement = [0,0,False]#信号を格納 [右,左,信号の有無]
        self.check = [0,0,0,0,0]#更新ステップにおける課題成功の有無[success,fail,conflict,other,other]
        self.check_eva = [0,0,0,0,0]#エピソードにおける課題成功の有無[success,fail,conflict,other,other]
        ## グラフに使用 ##
        self.x_oribit = []#ロボット軌道 
        self.y_oribit = []
        ## ロボットの設定 ##
        self.deg = [0,45,90,135,180,-45,-90,-135]#センサの角度
        self.dt = 0.3#[s]
        self.cp = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]#センサがあたっている、壁・障害物の座標
        self.cd=[-1,-1,-1,-1,-1,-1,-1,-1]#ロボットの中心と、センサの座標の距離(距離センサの値)
        self.px_init = 350#初期位置
        self.py_init = 700
        self.theta_init = math.radians(90)#角度[rad]
        ## 報酬の設定 ##
        self.reward_success = 5.0
        self.reward_fail = -5.0
        self.reward_clash = -2.0
        ## 壁の設定 ##
        #T字迷路 #
        self.walls = []
        self.walls.append([[50,50],[650,50],[650,1500],[50,1500]])#壁 0
        self.walls.append([[50,350],[250,350],[250,1500],[50,1500]])#壁 1
        self.walls.append([[450,350],[650,350],[650,1500],[450,1500]])#壁 2 
        ## 報酬線の設定 ##
        self.lines = []
        self.lines.append([[self.walls[1][1][0],50],[self.walls[1][1][0],self.walls[1][1][1]]])#変動する報酬左 0
        self.lines.append([[self.walls[2][0][0],50],[self.walls[2][0][0],self.walls[2][0][1]]])#変動する報酬右 1
        self.lines.append([[self.walls[1][1][0],self.py_init-50],[self.walls[2][0][0],self.py_init-50]])#信号を出す線
        self.lines.append([[self.walls[1][1][0],self.walls[2][1][1]],[self.walls[2][0][0],self.walls[2][0][1]]])#参考
        ## pygame ##
        if render == 1:
            pygame.init()#Pygameの初期化
            #self.im = pygame.image.load("robot.jpg")#robot画像読み込み
            self.im = pygame.image.load("robot.bmp")#robot画像読み込み
            self.im = pygame.transform.rotozoom(self.im, -90, 0.5)#画像の縮小と回転（,角度,倍率）
            self.clock = pygame.time.Clock()
            self.viewer = 0
            SCR_WIDTH,SCR_HEIGHT = 1500,1200#画面の大きさ
            self.screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))#画面を生成

    def reset(self):
        ### 状態を初期化し,初期の観測値を返す.エピソードごとにリセット. ###
        self.theta,self.px,self.py = self.theta_init,self.px_init,self.py_init
        self.check = [0,0,0,0,0]#更新ステップにおける課題成功の有無[success,fail,conflict,other,other]
        self.check_eva = [0,0,0,0,0]#エピソードにおける課題成功の有無[success,fail,conflict,success2,other]
        self.requirement = [0,0,False]#右、左、信号の有無
        self.x_oribit,self.y_oribit = [],[]
        self.reward = 0
        self.state = self.cd
        self.t = 0
        done = 0#終了判定
        return np.array(self.state), self.reward, done, {}

    
    def step(self, action):
        ### actionに応じて環境を変化させ、結果(state,reward)を返す ###
        ## 初期化 ##
        done = 0
        self.reward = 0
        self.requirement[2] = False #更新ステップで基準線を踏んだか
        self.check = [0,0,0,0,0]#更新ステップの課題成功の有無[success,fail,conflict,other,other]

        ## ロボットの動作 ##
        # actionを元に計算 #
        v,omega = calcurate_wheel(action)#ロボットの中心速度と回転角速度
        vx =  v*np.cos(self.theta)
        vy = -v*np.sin(self.theta)
        # px0:更新前の座標を保存する #
        px0 = 1.0*self.px
        py0 = 1.0*self.py
        # 更新 #
        self.px = self.px + vx*self.dt#座標更新x
        self.py = self.py + vy*self.dt#座標更新y
        self.theta = self.theta + omega*self.dt#角度の更新
          
        ## 衝突判定 ##
        robot_corners = calluculate_robot(self.px,self.py,25,self.theta)#ロボット四隅の座標を取得(四隅で判定)
        self.check[2],i_clash = check_intersection2(robot_corners,self.walls)#衝突判定(四隅で判定)
        #self.check[2],i_clash = check_intersection(px0,py0,self.px,self.py,self.walls)#衝突判定(中心で判定)
        self.cd, self.cp = get_distance_to_wall(self.px,self.py,self.theta,self.deg,self.walls)#距離センサの値を取得
        
        ## 左右の信号(初期刺激) ##
        if detect_pass(px0,py0,self.px,self.py,self.lines[2])[2] == 'toup':#信号線を踏んだら
            self.requirement[np.random.randint(0,2)] = 1##signal0は右、1は左
            self.requirement[2] = True
        
        ## 報酬(ゴール) ##
        if detect_pass(px0,py0,self.px,self.py,self.lines[0])[1] == 'toleft':#左上線を踏んだら
            if self.requirement[1] == 1:#信号左
                self.reward = self.reward_success
                self.check[0] = 1
            if self.requirement[0] == 1:#信号右
                self.reward = self.reward_fail
                self.check[1] = 1
            #self.requirement = [0,0,False]
        if detect_pass(px0,py0,self.px,self.py,self.lines[1])[1] == 'toright':#右上線を踏んだら
            if self.requirement[0] == 1:#信号右
                self.reward = self.reward_success
                self.check[0] = 1
            if self.requirement[1] == 1:#信号左
                self.reward = self.reward_fail
                self.check[1] = 1
            #self.requirement = [0,0,False]
    
        ## 報酬(衝突) ##
        if self.check[2] == 1: self.reward += self.reward_clash

        ## 成功失敗の評価に使用するもの ##
        if self.check[0] == 1:#成功
            self.check_eva[0] = 1
        if self.check[1] == 1:#失敗
            self.check_eva[1] = 1
        if self.check[2] ==1:#衝突
            self.check_eva[2] = 1

        ## 終了条件 ##
        if (self.check_eva[0] == 1) or (self.check_eva[1] == 1) or (self.check_eva[2] == 1):
            done = 1

        ## 更新 ##
        self.state = self.cd
        self.x_oribit.append(self.px)
        self.y_oribit.append(self.py)
        self.t += 1

        return np.array(self.state), self.reward, done, {}

    
    def render(self):
        ### 環境を可視化する ###
        ## 初期化 ##
        pygame.display.set_caption("robot_simulation")#タイトルバーに表示する文字
        self.screen.fill((255,255,255))#画面を白に塗りつぶす
        ### Walls ###
        for i in range(len(self.walls)):#壁を描く
            wall = self.walls[i]
            for j in range(len(wall)):
                if j==len(wall)-1:
                    pygame.draw.line(self.screen,(0,0,0),(wall[j][0],wall[j][1]),(wall[0][0],wall[0][1]),4)
                else:
                    pygame.draw.line(self.screen,(0,0,0),(wall[j][0],wall[j][1]),(wall[j+1][0],wall[j+1][1]),4)

        ## lines ##
        for k in range(len(self.lines)-2):
            line = self.lines[k+2]
            pygame.draw.line(self.screen,(0,200,100),(line[0][0],line[0][1]),(line[1][0],line[1][1]),2)
        if self.requirement[1] == 1:
            pygame.draw.line(self.screen,(255,165,0),(self.lines[0][0][0],self.lines[0][0][1]),(self.lines[0][1][0],self.lines[0][1][1]),2)#成功
            #pygame.draw.line(self.screen,(0,0,250),(self.lines[1][0][0],self.lines[1][0][1]),(self.lines[1][1][0],self.lines[1][1][1]),2)#失敗
        if self.requirement[0] == 1:
            pygame.draw.line(self.screen,(255,165,0),(self.lines[1][0][0],self.lines[1][0][1]),(self.lines[1][1][0],self.lines[1][1][1]),2)#成功
            #pygame.draw.line(self.screen,(0,0,250),(self.lines[0][0][0],self.lines[0][0][1]),(self.lines[0][1][0],self.lines[0][1][1]),2)#失敗
                        
        ## ロボット ##
        theta_deg = np.rad2deg(self.theta)
        #theta_deg = math.degrees(self.theta)
        new_im = pygame.transform.rotate(self.im,theta_deg)#ロボットの角度
        rect = new_im.get_rect()
        rect.center = (self.px,self.py)# TODO 中心位置を設定
        self.screen.blit(new_im,rect)#画面に描写

        ## 文字列 ##
        sysfont = pygame.font.SysFont(None,40)
        (text_x,text_y) = (800,0)
        text_dy = 30
    
        ## センサ値とセンサの線の表示 ##
        for g in range(len(self.deg)):
            pygame.draw.aaline(self.screen,(255,0,0),(self.px,self.py),(self.cp[g][0],self.cp[g][1]),2)#一番近い点との線を描く###
            dis=self.cd[g]
            messagese = sysfont.render("%4d: %6.2f" % (self.deg[g],round(dis,2)),True,(0,0,0))
            self.screen.blit(messagese,(text_x,text_y))
            text_y += text_dy
        messagese = sysfont.render("reward: %6.2f" % (self.reward),True,(0,0,0))
        self.screen.blit(messagese,(text_x,text_y))

        ## 更新設定 ##
        pygame.display.update()#画面更新
        self.clock.tick(50)#30fps


    def close(self):
        ### 環境を閉じて後処理をする ###
        if self.viewer: self.viewer.close()



    
