import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch.nn.functional as F
ct.set_cnml_enabled(False)
 #####float##########
input = torch.randn([1, 18, 2560]).half()
weight = torch.randn([7680, 2560]).half()
bias = torch.randn([7680]).half()

input_mlu = input.to("mlu")
weight_mlu = weight.to("mlu")
bias_mlu = bias.to('mlu')

print("input diff  {}".format(input.float()-input_mlu.float().cpu()))
# output = F.linear(input.float(),weight.float(), bias.float())
# output = F.linear(input,weight, bias)
# output_half = output.half()
# tensor在mlu上,可以使用.float()进行转换,但是结果可能不对z,只是有可能
# outputmlu = F.linear(input_mlu.float(),weight_mlu.float(), bias_mlu.float())
outputmlu = F.linear(input_mlu,weight_mlu, bias_mlu)
# outputmlu_half = outputmlu.half()
# # tensor是fp16的形式的,在ctr1.0上会报错.下面为例子
# # outputmlu = F.linear(input_mlu,weight_mlu, bias_mlu)
# # import pdb;pdb.set_trace()
# print(output-outputmlu.cpu())

# print(output_half-outputmlu_half.cpu())
'''

input = torch.randn([1, 18, 2560])
weight = torch.randn([7680, 2560])
bias = torch.randn([7680])
# import pdb;pdb.set_trace()
input_mlu = input.to("mlu")
weight_mlu = weight.to("mlu")
bias_mlu = bias.to('mlu')
output = F.linear(input,weight, bias)

outputmlu = F.linear(input_mlu,weight_mlu, bias_mlu)

print(output-outputmlu.cpu()) #相减结果为0

'''
