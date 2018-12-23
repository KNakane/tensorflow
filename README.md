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
┣ CNN : CNNで学習するディレクトリ  
┃   
┣ network  : DLのNetworkを構築するディレクトリ    
┃  
┣ rl  
┃  ┣ agents  : 強化学習用のAgent  
┃  ┣ atari  : atariを使用して強化学習  
┃  ┗ classic : 古典的なenvを使用して強化学習  
┃  
┣ segmentation : segmentationを行うディレクトリ  
┃  
┣ utility  
┃  
┗ README.md  