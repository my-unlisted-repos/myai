import torch
CPU = torch.device("cpu")
CUDA = torch.device('cuda')

CUDA_IF_AVAILABLE = CUDA if torch.cuda.is_available() else CPU
