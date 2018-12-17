tensorflow
==

# Overview
tensorflowの勉強用
CNNと強化学習について触れる

# Requirement
```
$ brew install pyenv
$ pyenv install 3.6.7
$ pyenv global 3.6.7
$ pip install numpy
$ pip install matplotlib
$ pip install requests
$ pip install tensorflow or pip install tensorflow-gpu
```


# Directory
ディレクトリ構造は以下の通り
tensorflow  
┣ CNN  
┃  ┣ data_load.py  
┃  ┣ model.py  
┃  ┣ optimizer.py  
┃  ┣ train.py  
┃  ┣ utils.py  
┃  ┗ README.md  
┃  
┣ rl  
┃ ┗ atari  
┃  　　　┣ atari_wrapper.py  
┃  　　　┣ cartpole_sample.py  
┃  　　　┣ cartpole_wrapper.py  
┃  　　　┣ display_as_gif.py  
┃  　　　┣ dqn.py  
┃  　　　┣ model.py  
┃  　　　┣ replay_memory.py  
┃  　　　┣ trainer.py  
┃  　　　┣ writer.py  
┃  　　　┗ README.md  
┃  
┗ README.md  