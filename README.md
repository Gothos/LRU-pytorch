# LRU-pytorch
An implementation of Linear Recurrent Units, by Deepmind, in Pytorch. LRUs are inspired by Deep State-Space Machines, particularly S4 and S5.

# Notes:
+ Since Pytorch does not have associative scans as of now, the Pytorch implementation will very likely be slower than a JAX implementation.
+ Complex tensors are still in beta in Pytorch and do not fully support .half(), so using torch.float16 is not advised.
+ Certain tensors are created on every forward pass. This is necessary only during training, and these tensors could be frozen to speed up inference.

# Installation:
```
$ pip install LRU-pytorch
```
# Usage:
```python
import torch

from LRU_pytorch import LRU

# Create a single Linear Recurrent Unit, that takes in inputs of size (batch_size, seq_length, 30) (or (seq_length, 30)), 
# with internal state-space variable of size 10, and returns outputs of (batch_size, seq_length, 50) (or (seq_length, 50)).

lru= LRU(
      in_features=30,
      out_features=50,
      state_features=10
      )

preds= lru(torch.randn([2,50,30])) # Get predictions
```
# Parameters:
```in_features```: int. The size of each timestep of the input sequence.

```out_features```: int. The size of each timestep of the output sequence.

```state_features```:int. The size of the internal state variable.

# Paper:
<a href='https://arxiv.org/abs/2303.06349'>Resurrecting Recurrent Neural Networks for Long Sequences</a>



