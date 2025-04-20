import torch

print("Torch version:", torch.__version__)

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

if cuda_available:
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

    # Test: przenieś tensor na GPU i wykonaj prostą operację
    x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
    y = x * 2
    print("Test tensor on CUDA:", y)
else:
    print("❌ CUDA not detected. Using CPU only.")
