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
N = audio.size # 37888
print("Total number of elements: {0} of type: {1}. Space required {2}"
      " kilobytes".format(audio.size, audio.dtype, 2 * audio.size / 2**10))

print("Size of convolution matrix N x N of type float64: {0} megabytes".format(
    8 * audio.size**2 / 2**20))

reduced_audio = audio[::4]
N = reduced_audio.size

print("Size of convolution matrix N x N of type float64: {0} megabytes".format(
    8 * reduced_audio.size**2 / 2**20))


def gen_toeplitz(N, alpha):
    numvec = np.linspace(0,N-1,N,dtype=np.int32)
    print("Length {} of type {}".format(numvec.size,numvec.dtype))
    print(numvec)
    Tij = np.sqrt(alpha/np.pi)*np.e**(-alpha*(numvec**2))
#    print(Tij)

    T = toeplitz(Tij,Tij)
    print(T)
    return T

gen_toeplitz(N,1/5)
T = gen_toeplitz(N,1/100)

print(np.matmul(T,reduced_audio))
