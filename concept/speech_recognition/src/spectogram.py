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

  def __init__(self, T_recording, N=512, T_eval=0.01):
    self.T_recording = T_recording
    self.N = N
    self.T_eval = T_eval

  def record(self):
    _waveform = sd.rec(self.T_recording*self.F_sample, samplerate=self.F_sample, channels=1)
    sd.wait()

def main():
  None

if __name__ == "__main__":
  main()
