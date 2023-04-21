import torch
import numpy as np
from common import Conv, Bottleneck, C2f, SPPF, C2fCustom

torch.manual_seed(0)

# t = torch.tensor(np.moveaxis(np.array([
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [2, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 6, 4]
#         ],
#         [
#             [1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 4], [4, 5, 6, 4], [4, 5, 5, 4]
#         ]
#     ]), [0, 1, 2], [2, 1, 0]))

# dim = 2

# print("EXAMPLE :")
# print(t)
# print(t.size())
# print("")
# t_s = torch.split(t, 4, dim)
# print("SPLITTED :")
# print(t_s[0])
# print(t_s[0].size())
# print("")
# res = torch.cat(t_s, dim)
# print(res.size())

# last = torch.cat((res, t_s[-1]), dim)
# print(last)
# print(last.size())

t = torch.full((1, 3, 8, 5), 2.)
# print(t)
print(t.size())

t_temp = Conv(3, 16, 3, 2, 1).forward(t)
t_temp = Conv(16, 32, 3, 2, 1).forward(t_temp)
print(t_temp)
print(t_temp.size())
print(C2f(32, 32, True, 1).forward(t_temp))
print(t_temp)
print(C2fCustom(32, 32, True, 1).forward(t_temp))