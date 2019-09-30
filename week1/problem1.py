from scipy.linalg import toeplitz
import numpy as np
import math
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from IPython.display import Audio
#%matplotlib notebook

# reading
rate, audio = wav.read("data/TMaRdy00.wav")

# plotting
plt.plot(audio)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("You wanna piece of me, boy?")
plt.show()

# playing
Audio(audio, rate=rate)

print(audio.dtype) # int16
print(audio.size) # 37888
print("Total number of elements: {0} of type: {1}. Space required {2}"
      " kilobytes".format(audio.size, audio.dtype, 2 * audio.size / 2**10))


from sys import getsizeof

print(getsizeof(audio))
print(audio.nbytes)
