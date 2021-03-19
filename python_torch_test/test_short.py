import torch

import torch.nn.functional as F

with torch.autograd.profiler.profile() as prof:
    input1 = torch.randn(1, 50, 2560)
    weight = torch.randn(7680,2560)
    bias = torch.randn(7680)
    m = F.linear(input1, weight, bias)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))



# int_tensor = torch.ones(1, dtype=torch.uint8)

# uint_tensor = torch.ones(1, dtype=torch.short)


# x = torch.randn((6,8))
# # print(x)
# index_list = np.array([[2,1],[1,0]])
# index_tensor = [torch.tensor(index_list[0]),torch.tensor(index_list[1])]

# with torch.autograd.profiler.profile(record_shapes=True) as prof:
#       uint_tensor*=int_tensor
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
# prof.export_chrome_trace("trace.json")

# x = torch.randn((1, 1), requires_grad=True)
# with torch.autograd.profiler.profile() as prof:
#     for _ in range(100):  # any normal python code, really!
#          y = x ** 2
#          y.backward()
 
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))


# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))

# with torch.autograd.profiler.profile() as prof:
#       x[index_list]
# #     x[0:2,[0]]
# #   #  for _ in range(100):  # any normal python code, really!
# #   #      x[index_tensor]
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# x[torch.tensor(index_list)]
# # print(x[0:2,0:1])
# rows = np.array([[0,0],[3,3]])
# cols = np.array([[0,2],[0,2]]) 
# y = x[rows,cols] 
# print(y)
