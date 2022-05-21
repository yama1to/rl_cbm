### いろいろとプロットするもの ###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
import csv
from sklearn.decomposition import PCA
import matplotlib.patches as patches


def mkdir():
    """
    ディレクトリを作成する
    """
    name = 'plot'
    if not os.path.exists(name):#ディレクトリがなかったら
        os.mkdir(name)
    #else:
        #shutil.rmtree(name)
        #os.mkdir(name)  

    name_list = ['x','r','u','q','all','orbit','histgram','gazou','pca']  
    for name_i in name_list:
        if not os.path.exists(name+'/'+name_i):#ディレクトリがなかったら
            os.mkdir(name+'/'+name_i)
        else:
            shutil.rmtree(name+'/'+name_i)
            os.mkdir(name+'/'+name_i)

def print_evaluation(list1,list2,num_episode):
    """
    報酬の到達率を100エピソードごとにターミナル表示
    """
    ## 引数は何か？ ##
    list1_name = '成功'
    list2_name = '成功２'

    for ep in range(int(num_episode/100)):
        a = sum(list1[ep*100:(ep+1)*100])
        b = sum(list2[ep*100:(ep+1)*100])
        print("エピソード{}~{}:{}={}回,{}={}回".format(ep*100,(ep+1)*100-1,list1_name,a,list2_name,b))
        print("{}の到達確率:{:.2f},{}の到達確率:{:.2f}".format(list1_name,a/100,list2_name,b/100))
    return a,b

def plot_hakohige(list1,list2,list3):
    """
    箱ひげ図を作成する(3つ)
    """
    ## リストは何を示しているのか(引数により変化する) ##
    list1_name = 'goal'
    list2_name = 'goal2'
    list3_name = 'nongoal'
    title = 'box plot'

    ## プロット ##
    points = (list1,list2,list3)
    fig, ax = plt.subplots()
    bp = ax.boxplot(points,showmeans=True)
    ax.set_xticklabels([list1_name,list2_name,list3_name])

    plt.title(title)
    plt.ylim([0,1])
    plt.grid()
    #plt.savefig('plot/hakohige.png')
    plt.savefig('plot/box_plot.eps')
    plt.show()
    plt.clf()
    plt.close()

def plot_hist(list1,num_episodes,seed):
    """
    棒グラフを作成
    num_episodeとnumberはmod0にする
    """
    number = 5 #ヒストグラムの棒の数
    title = 'histgram'
    list1_name = 'hist_success'
    range = num_episodes/number #一つの棒の範囲
    label = ["{}-{}".format(n*range,n*range+range) for n in range(number)]#ラベル横

    height = [sum(list1[c*range:c*range+range-1]) for c in range(number)]#縦軸計算
    left = [i for i in range(number)]#横軸
    plt.bar(left, height, tick_label=label, align="center")
    plt.title(title)
    plt.xlabel("episodes")
    plt.ylabel("Number of Successes")
    #plt.grid(True)
    plt.savefig("plot/histgram/"+"{}".format(seed)+".eps")
    plt.show()


def plot_csv(csv_list):
    """
    結果をcsvに読み出す
    """
    file = open('plot/new.csv', 'w')
    w = csv.writer(file)
    w.writerows(csv_list)
    file.close()

def plot_orbit(X,Y,episode):
    """
    エピソードの軌道をプロットする
    """
    filename = 'plot'+'/'+'orbit'+'/'+'orbit'+str(episode)
    plt.figure()
    plt.plot(X[episode],Y[episode],c=cm.hsv(episode/len(X)))
    plt.grid(True)
    plt.axis([50, 1050, 750, 50])#環境に依存する
    plt.xlabel("x(mm)")
    plt.ylabel("y(mm)")
    plt.savefig(filename)
    plt.clf()
    plt.close()


def plot_internal(episode,Q,U,R):
    """
    内部状態などをプロットする
    """
    # ### iternal X ###
    # plt.figure()
    # plt.plot(np.arange(len(X)),X)
    # plt.savefig('plot/x'+ '/'+str(episode)+'.eps')
    # plt.clf()
    # plt.close()

    ### firing R ###
    plt.plot(np.arange(len(R)),R)
    plt.savefig('plot/r'+ '/'+str(episode)+'.eps')
    plt.clf()
    plt.close()

    ### input U ###
    plt.figure()
    plt.plot(np.arange(len(U)),U)
    plt.savefig('plot/u'+ '/'+str(episode)+'.eps')
    plt.clf()
    plt.close()

    ### actionvalue Q ###
    plt.figure()
    plt.plot(np.arange(len(Q)),Q)
    plt.savefig('plot/q'+ '/'+str(episode)+'.eps')
    plt.clf()
    plt.close()

    ### all ###
    fig = plt.figure()
    # ax1 = fig.add_subplot(4, 1, 2)
    # ax1.plot(np.arange(len(X)),X)
    # ax1.set_ylabel("Reservoir state:x(t)")

    ax3 = fig.add_subplot(3, 1, 1)
    ax3.plot(np.arange(len(U)),U)
    ax3.set_ylabel("Input:u(t)")

    ax4 = fig.add_subplot(3, 1, 3)
    ax4.plot(np.arange(len(R)),R)
    ax4.set_ylabel("Friting rate:r(t)")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(np.arange(len(Q)),Q)
    #ax2.set_xlabel("gene")
    ax2.set_xlabel("Time t(s)")
    ax2.set_ylabel("Q-value(output):q(t)")
    plt.savefig('plot/all'+ '/'+str(episode)+'double.eps')
    plt.clf()
    plt.close()

def plot_orbit_all(x,y,e,num_episode):
    """
    複数の軌道を重ねてプロット
    全ての軌道を重ねてプロット
    最後のいくつかの軌道をプロット
    #x,y:軌道、e:seed、num_episode:エピソード最大
    """
    environment = [50, 650, 750, 50]
    filename = 'plot/gazou'
    #r = patches.Rectangle(xy=(50, 350), width=200, height=700, ec='#000000', fill=(125,125,125,0.5))#壁とか
    #r2 = patches.Rectangle(xy=(450, 350), width=200, height=700, ec='#000000', fill=(125,125,125,0.5))

    ### 複数エピソードずつ表示 ###
    number_row = 2#行
    number_colum = 4#列
    number_orbit = 200#重ねる軌道の数
    figs = plt.figure()
    rout_p = figs.subplots(number_row, number_colum)
    row = 0
    colum = 0
    for i in range(len(x)):
        rout_p[row][colum].plot(x[i],y[i],c=cm.hsv(i/len(x)))
        
        if (i+1)%number_orbit == 0:
            #rout_p[row][colum].grid()
            rout_p[row][colum].axis(environment)#環境に依存する
            colum += 1
        if (i+1)%(number_orbit*number_colum) == 0:
            row += 1
            colum = 0
    plt.axis(environment)#環境に依存する
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(filename+'/'+'separate'+str(e)+'.eps')
    plt.clf()
    plt.close()

    ### 全エピソードの軌道 ###
    plt.figure()
    for i in range(len(x)):
        plt.plot(x[i],y[i],c=cm.hsv(i/len(x)))
    plt.grid(True)
    plt.axis(environment)#環境に依存する
    plt.xlabel("x(mm)")
    plt.ylabel("y(mm)")
    plt.savefig(filename+'/' + 'all' + str(e)+'.eps')
    plt.clf()
    plt.close()

    
    ### 軌道(最終エピソード) ###
    for i in range(10):
        a = i+(num_episode-10)
        plt.figure()
        plt.plot(x[a],y[a],c=cm.hsv(a/len(x)))
        plt.grid(True)
        plt.axis(environment)#環境に依存する
        #plt.xlabel("x(mm)")
        #plt.ylabel("y(mm)")
        plt.savefig(filename+'/' + str(e) + str(a)+'.eps')
        plt.clf()
        plt.close()

def pre_pca(X_pca):
    '''
    リストを受け取りpcaをした後に三次元のリストを返す
    '''
    pca = PCA(n_components = 3,whiten = False)
    #pca = PCA()
    pca.fit(X_pca)

    x = pca.transform(X_pca)
    #print(len(x))
    print(x.shape)
    x_0 = [n for n in x[:,0]]
    x_1 = [n for n in x[:,1]]
    x_2 = [n for n in x[:,2]]
    print(pca.explained_variance_ratio_)#寄与率

    return x_0,x_1,x_2

def plot_pca(seed,x_0,x_1,x_2,x_pca_right):
    '''
    pca座標に変換したリスト(三次元)を信号の色ごとに散布図にする
    '''
    a = len(x_pca_right)#信号右は何個あるか
    #散布図バーじょん
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 50)      
    #ax.scatter(x1[:,0], x1[:,1], x1[:,2],alpha=0.8,s=1)
    #ax.scatter(x2[:,0], x2[:,1], x2[:,2],alpha=0.8,s=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.scatter(x_0[:a], x_1[:a], x_2[:a],alpha=0.8,s=1,c='orange')
    ax.scatter(x_0[a:], x_1[a:], x_2[a:],alpha=0.8,s=1,c='lime')
    #ax.plot(x_0,x_1,x_2)
    #plt.plot(x[:,0], x[:,1])
    plt.savefig('plot/pca/'+str(seed)+ '.png')#epsできない？？
    plt.show()
    plt.clf()
    plt.close()




