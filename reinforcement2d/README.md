### reinforcement2d
* RCを基にしたオンライン強化学習モデルによる移動ロボット制御
* 壁に囲まれた部屋のなかでゴール地点に近づくと報酬が得られる環境
* python main.py で動作する。
* reinforcement2cから学習更新式を変更。
* reinforcement2cからεと学習率の導出方法を報酬平均に基づくよう変更。そのため、報酬平均を求める変数なども追加している。
* その他ハイパーパラメータの変更はなし。

#### 準備
Ubuntu
Anaconda（3.x）
での動作を想定している。

pygameをインストールする。
$ pip install pygame --user
