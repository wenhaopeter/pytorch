import torch


# cand_ids = torch.randint(0,10000,(10,))
# print(cand_ids.dtype)
# print(cand_ids.shape)
scores = torch.Tensor([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])

a=scores[0:2]
# print(a)
# print(scores.dtype)
# print(scores.shape)
# scores[[1,2,3]]
# with torch.autograd.profiler.profile(record_shapes=True) as prof:
# print(scores[[1,2],[0,2]])
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
