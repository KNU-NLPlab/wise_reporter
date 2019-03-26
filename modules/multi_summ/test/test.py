import torch
import torch.nn as nn
import torch.nn.functional as F

import timeit

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

a = torch.Tensor([[1,2,3,4,5], [1,2,3,4,5]])
length = [5,4]

# rnn = torch.nn.RNN(10, 20, 1)
rnn = torch.nn.LSTM(10, 20, 1)


a = a.unsqueeze(2) * torch.ones(2,5,10)
a = torch.autograd.Variable(a)
print(a)
res1, res2 = rnn(a.transpose(0,1))
print(res1.t())

b = pack(a, length, batch_first=True)
m_bank, final = rnn(b)

m_bank = unpack(m_bank)[0].transpose(0,1)


print(m_bank)

print(res2)
print(final)