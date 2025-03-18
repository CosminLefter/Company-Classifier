import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device in Use:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))