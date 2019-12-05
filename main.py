# Code utilized
# - https://tinyurl.com/w7w93el (Stackoverflow)
# - https://tinyurl.com/ug7sr82 (Stackoverflow)
# - https://tinyurl.com/sm9sh2c (YouTube)
# - https://tinyurl.com/wnkzlem (Librosa Docs)


import scipy.io.wavfile as wave
import numpy as np
import matplotlib.pyplot as plt
import librosa.core as lib_core
import scipy.signal as sig

AUDIO_FILE = "./clerks_not_be_here.wav"


def main():
    rate, data = wave.read(AUDIO_FILE)
    data = np.array(data, dtype=np.float)
    d_lpc = lib_core.lpc(data, 16)
    time = np.linspace(0, len(data) / rate, num=len(data))
    d_hat = sig.lfilter([0] + -1*d_lpc[1:], [1], data)

    plt.figure(1)
    plt.title("Original Signal")
    plt.plot(time, data, color="r", label="Data")
    plt.plot(time, d_hat, color="b", label="LPC", linestyle="--")
    plt.show()

    wave.write("lpc_output.wav", rate, d_hat)


if __name__ == "__main__":
    main()
