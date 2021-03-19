import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch.nn.functional as F
ct.set_cnml_enabled(False)

val_float = torch.randn([1,2,3,4])

val_half = val_float.half()
print("float - half  {}".format(val_float-val_half))
# print("half  {}".format(val_half))
# print("half to float  {}".format(val_half.float()))
mlu_val_float = val_float.to("mlu")
mlu_val_half = val_half.to('mlu')

# print("mlf_half  {}".format(mlu_val_half.cpu()))
# print("mlf_half to mlu_float  {}".format(mlu_val_half.float().cpu()))

# print("mlu half -float   {}".format(mlu_val_half.cpu()-mlu_val_half.float().cpu()))

print("mlu_float - mlu_half {}  ".format(mlu_val_float.cpu()-mlu_val_float.half().cpu()))
print(mlu_val_half.device)

#结论,如果tensor在mlu上,使用.float()或者.half()确实能够改变数据类型,精度不受影响