tensorflow
==

# Overview
tensorflowの勉強用
CNNと強化学習について触れる

# Installation
```
$ brew install pyenv
$ pyenv install 3.6.7
$ pyenv global 3.6.7
$ pip install -r requirements.txt
```


# Directory
ディレクトリ構造は以下の通り
tensorflow 
┃  
┣ CNN  
┃  ┣ \_\_init\_\_.py  
┃  ┣ data_load.py     
┃  ┣ train.py  
┃  ┣ utils.py  
┃  ┗ README.md 
┃   
┣ network  
┃  ┣ \_\_init\_\_.py  
┃  ┣ lenet.py     
┃  ┣ model.py   
┃  ┗ module.py  
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
┣ utility  
┃  ┣ \_\_init\_\_.py 
┃  ┣ losses.py  
┃  ┣ utils.py  
┃  ┗ optimizer.py  
┗ README.md  