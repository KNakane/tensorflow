rl
====

# Overview
強化学習を行うフォルダ  
使用できるEnvironmentは以下の通り
- [Open AI gym](https://gym.openai.com/) : CartpoleやBreakout, Invaderなど
- [pygame](https://pygame-learning-environment.readthedocs.io/en/latest/)  : python上でゲームの環境が作成できる  
- [Roboschool](https://github.com/openai/roboschool) : mujocoと同様の環境を用意することができる

# Description
[All env](https://medium.com/@researchplex/openai-gym-environment-full-list-8b2e8ac4c1f7)にgymで使用できるenvironmentの一覧が記載されている  
## 使用できるAgent
|Agent|wrapper|備考|
|:----:|:-----:|:----:|
|DQN|atari,cartpole,pygame|-|
|DDQN|atari,cartpole,pygame|-|
|Rainbow|atari,cartpole,pygame|-|
|Ape-X||未実装|
|A3C||未実装|
|A2C||未実装|
|DDPG|continuous,roboschool|-|
|TD3|continuous,roboschool|-|


# Usage
## Build Network
- 各wrapper内にset_model関数があるため、そこにNetworkの構造を作成する
- 作成方法は[README.md](../CNN/README.md)のBuildNetworkを参照
- 注意点：conv -> fcの時には、その間にflatが必要（入力次元を下げるため）

## Wrapper
各environmentに合わせて、wrapperを用意しており、そこから実行するようにしている
- atari/atari_wrapper.py : atariを使用して強化学習を行う
- classic/cartpole_wrapper.py : cartpoleで強化学習を行う
- classic/continuous_wrapper.py : 連続値を扱う強化学習を行う
- pygame/pygame_wrapper.py : pygameで強化学習を行う

## Learning
各wrapperの使い方は  

- [atari usage](atari/README.md)
- [classic usage](classic/README.md)
- [pygame usage](pygame/README.md)

を参照のこと


## reference
- [multi-step learning](https://github.com/belepi93/pytorch-rainbow/blob/master/train.py)