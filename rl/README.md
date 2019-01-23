rl
====

# Overview
[Open AI gym](https://gym.openai.com/)を使って、強化学習を行う  
masterブランチではDefine and Runで実行するように作成していたが、devブランチではEagerモードで作成していく

# Description
[All env](https://medium.com/@researchplex/openai-gym-environment-full-list-8b2e8ac4c1f7)にgymで使用できるenvironmentの一覧が記載されている

# Usage
## Build Network
- 各wrapper内にset_model関数があるため、そこにNetworkの構造を作成する
- 作成方法は[README.md](../CNN/README.md)のBuildNetworkを参照
- 注意点：conv -> fcの時には、その間にflatが必要（入力次元を下げるため）

## Wrapper
各environmentに合わせて、wrapperを用意しており、そこから実行するようにしている
- atari/atari_wrapper.py : atariを使用して強化学習を行う
- classic/cartpole_wrapper.py : cartpoleで強化学習を行う
- classic/cacher_wrapper.py : cacherで強化学習を行う

## Learning
各wrapperの使い方は  

- [classic usage](classic/README.md)
- [atari usage](atari/README.md)

を参照のこと