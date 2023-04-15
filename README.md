# LRU-pytorch
An implementation of Linear Recurrent Units, by Deepmind, in Pytorch. LRUs are inspired by Deep State-Space Machines, particularly S4 and S5.

# Notes:
Since Pytorch does not have associative scans as of now, the Pytorch implementation will very likely be slower than a JAX implementation.

# Installation:
```
$ pip install LRU-pytorch
```
# Usage:
```python
import torch

from LRU_pytorch import LRU

# Create a single Linear Recurrent Unit, that takes in inputs of size (batch_size, seq_length, 30)
# (or (seq_length, 30)), with internal state-space variable of size 10, and returns outputs of 
# (batch_size, seq_length, 50) (or (seq_length, 50)).

LRU=(30,50,10) 
```


# Paper:
<a href='https://arxiv.org/abs/2303.06349'>Resurrecting Recurrent Neural Networks for Long Sequences</a>



