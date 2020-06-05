import torch.nn as nn
import torch
l = nn.Linear(2, 5)

v = torch.FloatTensor([1, 2])

l(v)

list(l.parameters())

print(l.state_dict())

s  = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)

s

s(torch.FloatTensor([[1, 2]]))