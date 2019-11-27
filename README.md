# seqGAN
Seq-GAN is a unique approach which models the data generator as a stochastic policy in reinforcement learning to solve the problem. The Reinforcement Learning reward signal comes from the GAN discriminator judged on a complete sequence, and is passed back to the intermediate state-action steps using Monte Carlo search.
