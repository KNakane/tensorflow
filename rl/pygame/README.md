Classic Control
====

# Overview
tensorflowで強化学習を学ぶ用  

# Description
- pygame_wrapperでPixelCopter-v0を行う場合は、```src/ple/ple/__init__.py```と```src/ple/ple/pixelcopter.py```内のクラス名を変更する必要があるため注意

# Environment
- Catcher-v0
- FlappyBird-v0
- Pong-v0
- PixelCopter-v0
- MonsterKong-v0
- PuckWorld-v0
- RaycastMaze-v0
- Snake-v0
- WaterWorld-v0

# Usage
```bash
python pygame_wrapper.py --env (environment名) \
                         --agent (agent名) \
                         --n_episode (episode数) \
                         --network (network名) \
                         --step (step数) \
                         --batch_size (batchサイズ) \
                         --multi_step (multi_step数) \
                         --n_warmup (warmupまでの回数) \
                         --model_update (modelを更新する間隔) \
                         --render (renderするかどうか) \
                         --priority (PER) \
                         --lr (learning rate) \
                         --opt (optimizer)
```

# Result
## Catcher
|Agent|50回学習|100回学習|
|:--:|:--:|:--:|
|DQN|<img src="../../sample_results/rl/catcher/DQN_50.gif" width="100%">|<img src="../../sample_results/rl/catcher/DQN_100.gif" width="100%">|
|DDQN|<img src="../../sample_results/rl/catcher/DDQN_50.gif" width="100%">|<img src="../../sample_results/rl/catcher/DDQN_100.gif" width="100%">|
|DQN|||
|DDQN + Dueling|<img src="../../sample_results/rl/catcher/DDQN_duel_50.gif" width="100%">|<img src="../../sample_results/rl/catcher/DDQN_duel_100.gif" width="100%">|
|DDQN + Dueling + priority + multi_step|<img src="../../sample_results/rl/catcher/DDQN_priority_multi_duel_50.gif" width="100%">|<img src="../../sample_results/rl/catcher/DDQN_priority_multi_duel_100.gif" width="100%">|
