import torch
import torchvision
import easyocr
import time

start = time.time()
print(torch.__version__)
print(torchvision.__version__)
reader = easyocr.Reader(['en'])
stop = time.time()
print(stop-start)
