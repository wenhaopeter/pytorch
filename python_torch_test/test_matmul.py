import torch
import torch_mlu.core.mlu_model as ct
ct.set_cnml_enabled(False)
ct.set_quantized_bitwidth(16)

mat1 = torch.randn([ 18, 80], dtype=torch.float).half()
mat2 = torch.randn([ 80, 18], dtype=torch.float).half()

# out_cpu = torch.matmul(mat1, mat2)
mat1=mat1.to(ct.mlu_device())
mat2=mat2.to(ct.mlu_device())
# out_mlu = torch.matmul(mat1.to(ct.mlu_device()), mat2.to(ct.mlu_device()))
mat1.mm(mat2)
