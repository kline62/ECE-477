import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write, read

class spectrogram:
  # Attributes
  # Public
  F_sample = 44100 # samle rate in Hz
  T_recording = None # duration of recording in seconds
  N = None # points in the DFT
  T_eval = None # Interval to evaluate DFT at in seconds
  # Private
  _waveform = None # Stored waveform
  _spectro = None # The spectrogram matrix

  def __init__(self, T_recording, N=512, T_eval=0.0001):
    self.T_recording = T_recording
    self.N = N
    self.T_eval = T_eval

  def record(self):
    print("RECORDING...")
    self._waveform = sd.rec(self.T_recording*self.F_sample, samplerate=self.F_sample, channels=1)
    sd.wait()
    print("DONE")

  def eval(self):
    if self._waveform is None:
      print("ERROR: no waveform has been loaded into the spectrogram object")
      return
    # Initialize the spectrogram
    self._spectro = np.matrix(np.zeros((int(self.T_recording/self.T_eval), self.N)))
    # Evaluate the dft at timed intervals
    step = int(self.T_eval*self.F_sample)
    for i in range(int(self.T_recording/self.T_eval)):
      dft = np.fft.fft(self._waveform[i*step:i*step+self.N],axis=1)
      if np.shape(dft)[0] != self.N:
        continue
      self._spectro[i,:] = dft.T

  def plot(self):
    if self._spectro is None:
      print("ERROR: spectrogram has not been evaluated")
      return
    plt.pcolormesh(self._spectro.T)
    #plt.plot(self._waveform)
    plt.show()

def main():
  s = spectrogram(4)
  s.record()
  s.eval()
  s.plot()

if __name__ == "__main__":
  main()
