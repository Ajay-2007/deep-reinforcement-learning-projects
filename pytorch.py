import torch
import numpy as np

a = torch.FloatTensor(3, 2)

print(a)

print(a.zero_())

torch.FloatTensor([[1, 2, 3], [3, 2, 1]])
n = np.zeros(shape=(3, 2))
print(n)

b = torch.tensor(n)
print(b)

n = np.zeros(shape=(3, 2), dtype=np.float32)
n

torch.tensor(n)

n = np.zeros(shape=(3, 2))
torch.tensor(n, dtype=torch.float32)


# scalar tensors

a = torch.tensor([1, 2, 3])
a
s = a.sum()
s
s.item()
torch.tensor(1)

# GPU tensors
a = torch.FloatTensor([2, 3])
a
ca = a.to("cuda"); ca

# Tensors and Gradients

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum*2).sum()
v_res
v1.is_leaf, v2.is_leaf
v_sum.is_leaf, v_res.is_leaf

v1.requires_grad
v2.requires_grad
v_sum.requires_grad
v_res.requires_grad
v_res.backward()
v1.grad
v2.grad