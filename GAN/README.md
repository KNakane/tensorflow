Generative Adversarial Network
--
# vanilla GAN(GAN)
```bash
$ python train.py --network GAN --opt Adam 
```

# DCGAN
```bash
$ python train.py --network DCGAN --opt Adam 
```
![代替テキスト](../sample_results/GAN/DCGAN.png)

# LSGAN
```bash
$ python train.py --network LSGAN --opt Adam 
```
![代替テキスト](../sample_results/GAN/LSGAN.png)

# Spectral Normalization GAN(SNGAN)
```bash
$ python train.py --network SNGAN --opt Adam
```
![代替テキスト](../sample_results/GAN/SNGAN.png)

# Wasserstein GAN(WGAN)
```bash
$ python train.py --network WGAN --opt RMSprop --n_disc_update 5 --lr 0.00005
```
![代替テキスト](../sample_results/GAN/WGAN.png)

# WGAN-GP  
```bash
$ python train.py --network WGANGP --opt RMSprop --n_disc_update 5 --lr 0.00005
```
![代替テキスト](../sample_results/GAN/WGANGP.png)

# Conditional GAN(CGAN)
- 各GANを実行する際に```--conditional```を追加するとconditional GANになる
![代替テキスト](../sample_results/GAN/cGAN.png)

# ACGAN
![代替テキスト](../sample_results/GAN/ACGAN.png)