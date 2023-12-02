import numpy as np
from PIL import Image

# Flowfields
coarse = np.array(Image.open('coarse.png'))
fine = np.array(Image.open('fine.png'))
# Crop these a bit to fit the output of the network
coarse = coarse[3:-2, 2:-2]
fine = fine[3:-2, 2:-2]

# Output from neural network
strength0 = np.array(Image.open('output/output_strength0.2_001.png'))
strength1 = np.array(Image.open('output/output_strength0.5_004.png'))
strength2 = np.array(Image.open('output/output_strength0.8_001.png'))

# Errors
def rms(a, b):
    return np.sqrt(np.mean((a - b)**2))
etrue = rms(coarse, fine)
e0 = rms(strength0, fine)
e1 = rms(strength1, fine)
e2 = rms(strength2, fine)

print("Error between coarse and fine flowfield:")
print(etrue)
print("Error between strength0 and fine flowfield:")
print(e0)
print("Error between strength1 and fine flowfield:")
print(e1)
print("Error between strength2 and fine flowfield:")
print(e2)
