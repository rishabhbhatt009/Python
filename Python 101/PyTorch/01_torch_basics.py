# Import ----------------------------------------------------------------------------------------
import torch
import numpy as np

print('Torch Imported', '-' * 50)

# Creating Tensor -------------------------------------------------------------------------------
# lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#
# t1 = torch.tensor(data=lst)
# t2 = torch.as_tensor(data=lst, dtype=torch.int8)
# t3 = torch.ones(size=(3, 3), dtype=torch.float)
# t4 = torch.full(size=(3, 3), fill_value=9)

# t5 = torch.range(1, 10, 1) will be deprecated
# t6 = torch.arange(1, 10, 1)
# t7 = torch.from_numpy(np.arange(1, 10, 1))

# t8 = torch.linspace(0, 100, 11)
# t9 = torch.logspace(1, 10, 10, base=2)

# Math operations -------------------------------------------------------------------------------
# t_add = t3 + t4
# t_add = t3.add(t4)
# t3.add_(t4)

# copy1 = t1.view(3,3)
# copy2 = torch.tensor(t1)
# print(t1, copy1, copy2,
#       sep='\n')
#
# t1[0][0] = 999
#
# print(t1, copy1, copy2,
#       sep='\n')

# Autograd --------------------------------------------------------------------------------------
# x1 = torch.ones(size=(3, 3), requires_grad=True)
# x2 = torch.ones(size=(3, 3), requires_grad=True)
# y1 = x1 + 2
# y2 = x2.mean()
#
# print(x1,
#       x1.data, x1.grad,
#       y1,
#       sep='\n')
#
# v = torch.rand(size=(3, 3), dtype=torch.float32)
# y1.backward(v)
# y2.backward()
#
# print()
# print(x1,
#       x1.data, x1.grad,
#       y1,
#       sep='\n')

# Back Propagation -------------------------------------------------------------------------------
# x = torch.tensor(1.)
# y = torch.tensor(2.)
#
# wt = torch.tensor(1., requires_grad=True)
#
# # Fwd Pass and computing loss
# y_hat = wt * x
# loss = (y_hat - y) ** 2
# print(loss, wt, wt.grad, sep='\n')
#
# # Bwd Pass
# loss.backward()
# print(loss, wt, wt.grad, sep='\n')

# update weights
# repeat process (fwd and bwd pass)

