rl
====

# Overview
tensorflowで強化学習を学ぶ用  
使用するデータはatari

# Description
- OpenAI gymのatariを用いて強化学習を行う
- mujocoのライセンスは未取得のため、使えない

# Usage
```bash
python atari_wrapper.py --env (game名) \
                        --n_episode (episode数) \
                        --step (step数) \
                        --batch_size (batchサイズ) \
                        --n_warmup (warmupまでの回数) \
                        --model_update (modelを更新する間隔) \
                        --render (renderするかどうか) \
                        --lr (learning rate) \
                        --opt (optimizer)
```
# Sample Result
```bash

```