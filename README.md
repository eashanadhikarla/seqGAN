# SeqGAN in Pytorch

SeqGAN adapts GAN for sequential generation. It regards the generator as a policy in reinforcement learning and the discriminator is trained to provide the reward. To evaluate unfinished sequences, Monto-Carlo search is also applied to sample the complete sequences.

This project is implemented by [Eashan Adhikarla](https://sites.google.com/eashanadhikarla) and reviewed by [Prof. Xie Sihong](http://www.cse.lehigh.edu/~sxie/projects.html).

## Descriptions
This project includes a [[Pytorch](https://github.com/pytorch)] implementation of **SeqGAN** proposed in the paper [[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)] by Lantao Yu et al. at Shanghai Jiao Tong University and University College London.

<p align="center">
    <img src="https://github.com/LantaoYu/SeqGAN/raw/master/figures/seqgan.png">
</p>

## Problem Statement
Due to generator differentiation problem, Generative Adversarial Networks have faced serious issues with updating the policy gradient. Seq-GAN is a unique ap- proach which models the data generator as a stochastic policy in reinforcement learning to solve the problem.

Past RNN (Recurrent Neural Network) training methods have shown some good results such as, pro- vided the previously observed token, maximizing the log predictive probability for each true token in the training sequence. However, this solution1 suffers from an exposure bias issue during infer- ence. This discrepancy between training and inference yielded errors that can accumulate quickly along the generated sequence. In this, they suggested scheduled sampling (SS) to address this issue by including some synthetic data during the phrase of learning. Yet later SS has proved to be an incoherent learning technique to be implemented.

Another approach which was used very popularly is to built the loss function of the entire generated sequence instead of each transition. However, this approach did not show up to the mark results for some of the real life complex examples like music, dialog, etc.

The discriminator is well trained in distinguishing between the actual image and the artificial image even in the Generative Adversarial Network (GAN), but GAN performs poorly with the discrete token values since the guidance is too small to cause a change in the restricted dictionary space. The paper suggests using Reinforcement Learning to train the generator portion of GAN.

## Prerequisites

- Python 2.7
- [Pytorch 1.2.0](https://pytorch.org/)
- pretty-midi (only for music dataset)

## Usage

### Datasets

- A randomly initialized LSTM is used to simulate a specific distribution.
- A music dataset contains multiple Nottingham Songs.

### Training LSTM 

- Run 
```
python2 main.py --pretrain_g_epochs 2000 --total_epochs 0 --log_dir logs/train/pure_pretrain --eval_log_dir logs/eval/pure_pretrain
```
to have a baseline that is trained with pure MLE loss for 2000 iterations.
- Run 
```
python2 main.py --pretrain_g_epochs 1000 --total_epochs 1000 --log_dir logs/train/pretrain_n_seqgan  --eval_log_dir logs/eval/with_seqgan
```
to train the model with first pretraining loss and then SeqGAN's loss.
- Run 
```
tensorboard --logdir logs/eval/
```
and open your browser to check the improvement that SeqGAN provided.

### Music generation

- Run `bash train_nottingham.sh` to train the model. Check data/Nottingham/\*.mid for generations. The songs will be updated every 100 epochs.

## Results

<p align="center">
    <img src="figures/SeqGAN.png">
</p>
In this figure, the blue line is Negative Log Likelihood(NLL) of purely using supervised learning (MLE loss) to train the generator, while the orange one is first using MLE to pretrain and then optimizing the adversarial loss. Two curves overlap with each other at the beginning since the same random seed is used.  After using SeqGAN's loss, the NLL drops and converges to a smaller loss, which indicates that the generated sequences match the distribution of the randomly intialized LSTM better. 

## Related works

[Yu, Lantao, et al. "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient." AAAI. 2017.](https://arxiv.org/abs/1609.05473)
