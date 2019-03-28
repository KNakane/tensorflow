Classic Control
====

# Overview
tensorflowで強化学習を学ぶ用  

# Description
- OpenAI gymを用いて強化学習を行う
- mujocoのライセンスは未取得のため、使えない
- pygame_wrapperでPixelCopter-v0を行う場合は、```src/ple/ple/__init__.py```と```src/ple/ple/pixelcopter.py```内のクラス名を変更する必要があるため注意

# Usage
## cartpole
```bash
python hogehoge_wrpper.py --env (environment名) \
                          --n_episode (episode数) \
                          --step (step数) \
                          --batch_size (batchサイズ) \
                          --n_warmup (warmupまでの回数) \
                          --model_update (modelを更新する間隔) \
                          --render (renderするかどうか) \
                          --lr (learning rate) \
                          --opt (optimizer)
```

# Result
## cartpole_wrapper
|Agent|100回学習|150回学習|
|:--:|:--:|:--:|
|DQN|![代替テキスト](../../sample_results/rl/cartpole/DQN_100.gif)|![代替テキスト](../../sample_results/rl/cartpole/DQN_150.gif)|
|DDQN|![代替テキスト](../../sample_results/rl/cartpole/DDQN_100.gif)|![代替テキスト](../../sample_results/rl/cartpole/DDQN_150.gif)|
|DQN + Dueling|![代替テキスト](../../sample_results/rl/cartpole/DQN_Duel_100.gif)|![代替テキスト](../../sample_results/rl/cartpole/DQN_Duel_150.gif)|
|DDQN + Dueling|![代替テキスト](../../sample_results/rl/cartpole/DDQN_Duel_100.gif)|![代替テキスト](../../sample_results/rl/cartpole/DDQN_Duel_150.gif)|
## Pendulum
|Agent|100回学習|150回学習|
|:--:|:--:|:--:|
|DDPG|![代替テキスト](../../sample_results/rl/pendulum/DDPG_50.gif)|![代替テキスト](../../sample_results/rl/pendulum/DDPG_100.gif)|
|TD3|![代替テキスト](../../sample_results/rl/pendulum/TD3_50.gif)|![代替テキスト](../../sample_results/rl/pendulum/TD3_100.gif)|

## Half Cheetah
|Agent|0回学習|200回学習|
|:--:|:--:|:--:|
|DDPG+PER+multi_step|![代替テキスト](../../sample_results/rl/half_cheetah/DDPG_PER_multi_0.gif)|![代替テキスト](../../sample_results/rl/half_cheetah/DDPG_PER_multi_200.gif)|