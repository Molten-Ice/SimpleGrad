# SimpleGrad

Minimal autograd implementation expanding micrograd to work with matrix operations and tensors.

Based on:
1. [karpathy's micrograd](https://github.com/karpathy/micrograd)
2. Michael Nielsen's [Neural Networks and Deep Learning book](http://neuralnetworksanddeeplearning.com/)
   - [Original repo](https://github.com/mnielsen/neural-networks-and-deep-learning)
   - [Python 3 fork](https://github.com/unexploredtest/neural-networks-and-deep-learning.git)

GPU used: NVIDIA GeForce GTX 1650 Ti Max-Q GPU


Baseline:
[2900/3000]: 474 / 1000 correct
Epoch 0: 4939 / 10000, took 1.95 seconds

Adding Xavier/Glorot (for Sigmoid activation)

[2900/3000]: 798 / 1000 correct
Epoch 0: 8146 / 10000, took 2.57 seconds
