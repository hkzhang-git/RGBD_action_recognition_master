import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

conv = nn.Conv2d(3, 1, 3, stride=2)
state_dict = conv.state_dict()
input = Variable(torch.from_numpy(np.random.randint(0, 255, (1, 3, 5, 5))).float())
k=torch.from_numpy(np.random.random((1, 3, 3, 3)))
state_dict['weight']=k
conv.load_state_dict(state_dict)
output = conv(input)
