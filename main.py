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
from functools import reduce

AUDIO_FILE = "./clerks_not_be_here.wav"


def main():
    rate, data = wave.read(AUDIO_FILE)
    # Convert data to float so that it is compatible with
    # librosa lpc algorithm
    data = np.array(data, dtype=np.float)
    d_lpc = lib_core.lpc(data, 16)
    # Get the time domain values to plot on the x-axis
    time = np.linspace(0, len(data) / rate, num=len(data))
    # Rebuild signal using the LPC coefficients
    d_hat = sig.lfilter([0] + -1*d_lpc[1:], [1], data)

    # Graph the lpc signal on top of the original signal
    plt.figure(1)
    plt.title("Original Signal")
    plt.plot(time, data, color="r", label="Data")
    plt.plot(time, d_hat, color="b", label="LPC", linestyle="--")
    plt.show()

    # Need to change d_hat to be integers so that the wav file may be read
    d_hat = np.array(d_hat, dtype=np.int16)
    data = np.array(data, dtype=np.int16)

    # Create new wav file
    # Volume is very low so we multiply the output values by a factor
    wave.write("lpc_output.wav", rate, d_hat * 20)

    # Find the excitation signal
    excitation = data - d_hat
    plt.figure(2)
    plt.title("Excitation Signal")
    plt.plot(time, excitation, color="r")
    plt.show()

    # Needs to be > 16 bit number to prevent overflow
    excitation = np.array(excitation, dtype=np.int64)
    print(
        f"Sum of absolute value of excitation signal: {reduce(lambda a,b: a + abs(b), excitation)}")


if __name__ == "__main__":
    main()
