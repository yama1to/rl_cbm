## data20220518_101535_main_HRC_agent_q
### Common config
```
hostname: pc-34
dir_path: data/data20220518_101535_main_HRC_agent_q
Report  : data20220518_101535_main_HRC_agent_q.md
Test    : None
Exe     : python3 main_HRC_agent_q.py 
parallel: 12
```
### Default Config
```
 columns:None
     csv:None
      id:None
    plot:0
    seed:0
train_episode_num:20
test_episode_num:100
show_env:0
is_minus:1
report_interval_train:50
report_interval_test:50
max_step_num:20
      NN:256
      Nu:16
      Nh:500
      Ny:4
    Temp:1.0
      dt:0.00390625
 alpha_i:0.6
 alpha_r:0.9
 alpha_b:0.0
 alpha_s:0.1
  beta_i:0.3
  beta_r:0.8
    ep_2:1
  ep_ini:0.1
  ep_fin:0.0
   eta_2:0.1
 eta_ini:0.01
 eta_fin:0
gamma_wout:0.9
k_greedy:0.005
test_prob:0
```
### Optimization 
Configuration:  
```
id      : 0.000000
seed    :
beta_i  : 0.860000[ 0.000000, 1.000000](2)
alpha_i : 0.400000[ 0.000000, 1.000000](2)
beta_r  : 0.580000[ 0.000000, 1.000000](2)
alpha_r : 0.080000[ 0.000000, 1.000000](2)
alpha_s : 0.980000[ 0.000000, 2.000000](2)
target: test_prob 
```
Start:20220518_101535  
Done :20220518_101635  
Optimization result:  
```
plot    : 0.000000
seed    : 0.000000
train_episode_num:20.000000
test_episode_num:100.000000
max_step_num:20.000000
show_env: 0.000000
is_minus: 1.000000
NN      :256.000000
Nu      :16.000000
Nh      :500.000000
Ny      : 4.000000
Temp    : 1.000000
alpha_i : 0.400000
alpha_r : 0.080000
alpha_s : 0.980000
beta_i  : 0.860000
beta_r  : 0.580000
ep_ini  : 0.100000
ep_fin  : 0.000000
ep_2    : 1.000000
eta_ini : 0.010000
eta_fin : 0.000000
eta_2   : 0.100000
gamma_wout: 0.900000
k_greedy: 0.005000
test_prob: 0.000000
TARGET  : 0.000000
```
