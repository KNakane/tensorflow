tensorflow
==

# Overview
tensorflowの勉強用  
CNNと強化学習などについて触れる  
## [AutoEncoder](./AutoEncoder/README.md)
AutoEncoderによる画像生成を行う  
- AutoEncoder(AE)  
- Denosing AutoEncoder(DAE)  
- Variable AutoEncoder(VAE)  
- Conditional Variable AutoEncoder(CVAE)  

## [CNN](./CNN/README.md)
MonitoredTrainingSessionを使用し、Graph Modeで学習を行っていく  
tensorboardでgraphや各数値を確認することが出来ます  
ネットワーク構造は自分で作成することができるが、以下のネットワークはすでに作成済み
- LeNet
- VGG
- ResNet
- ResNext
- DenseNet
- SENet

## [GAN](./GAN/README.md)
GANによる画像生成ができる
- vanilla GAN(GAN)
- DCGAN
- Wasserstein GAN(WGAN)
- WGAN-GP  
- Conditional GAN(CGAN)

## [強化学習](./rl/README.md)
tensorflowのEager Modeを使用している  
[Eagerモード参考URL](https://www.hellocybernetics.tech/entry/2018/12/04/231714)
 

## dataset
使用できるデータセットは以下の通り
- [mnist](http://yann.lecun.com/exdb/mnist/)
- [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [kuzushiji-mnist](https://github.com/rois-codh/kmnist)

使い方などは各ディレクトリのREADME.mdを見てください 

# Installation
```
$ brew install imagemagick
$ brew install pyenv
$ pyenv install 3.6.7
$ pyenv global 3.6.7
$ git clone https://github.com/KNakane/tensorflow.git
$ pip install -r requirements.txt
$ pip install tensorflow or pip install tensorflow-gpu
```

# Usage
全てのプログラムは基本的にここのディレクトリから呼び出して実行する必要がある
```bash
$ cd tensorflow
$ python path/to/dir.py --args
```
実行結果は```results```に格納される

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
┃  ┣ classic : 古典的なenvを使用して強化学習  
┃  ┗ pygame : pygameを使用して強化学習  
┃  
┣ segmentation : segmentationを行うディレクトリ  
┃  
┣ utility  
┃  
┗ README.md  