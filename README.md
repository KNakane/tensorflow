tensorflow
==

# Overview
tensorflowの勉強用  
CNNと強化学習について触れる  
強化学習に関しては、Eagerモードを使用するように作成していく(未作成部分多数)  
使い方などは各ディレクトリのREADME.mdを見てください  
[Eagerモード参考URL](https://www.hellocybernetics.tech/entry/2018/12/04/231714)  

# Installation
```
$ brew install pyenv
$ pyenv install 3.6.7
$ pyenv global 3.6.7
$ git clone https://github.com/KNakane/tensorflow.git
$ pip install -r requirements.txt
$ pip install tensorflow or pip install tensorflow-gpu
```


# Directory
ディレクトリ構造は以下の通り  
tensorflow   
┃  
┣ AutoEncoder : AutoEncoderが使用できるディレクトリ    
┃  
┣ CNN : CNNで学習するディレクトリ  
┃   
┣ GAN : GANを行うディレクトリ  
┃   
┣ dataset  : dataset取得用ディレクトリ  
┃   
┣ network  : DLのNetworkを構築するディレクトリ    
┃  
┣ optuna : PFNが作成したハイパーパラメータチューニングを行うディレクトリ    
┃  
┣ rl  
┃  ┣ agents  : 強化学習用のAgent  
┃  ┣ env    : pygame用のenvironment構築  
┃  ┣ atari  : atariを使用して強化学習  
┃  ┗ classic : 古典的なenvを使用して強化学習  
┃  
┣ segmentation : segmentationを行うディレクトリ  
┃  
┣ utility  
┃  
┗ README.md  
