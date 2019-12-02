# SeqGAN

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

## Requirements

- Python 3.6 or >
- colorama
- numpy      >= 1.12.1
- [tensorflow >= 1.5.0] (https://www.tensorflow.org/)
- scipy      >= 0.19.0
- nltk       >= 3.2.3

### Datasets

- A randomly initialized LSTM is used to simulate a specific distribution.
- Obama Speech Dataset.
- Chinese Poem Dataset.

## Related works

[Yu, Lantao, et al. "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient." AAAI. 2017.](https://arxiv.org/abs/1609.05473)
