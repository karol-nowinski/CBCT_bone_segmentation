import multiprocessing
import torch

# Skrypt sprawdzający liczbę możliwych rdzeni CPU - do wykorzystywania później w data loaderze

print("🧠 Liczba rdzeni CPU:", multiprocessing.cpu_count())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())