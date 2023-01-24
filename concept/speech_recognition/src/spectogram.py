import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import sounddevice as sd
from scipy.io.wavfile import write, read
from sklearn.naive_bayes import GaussianNB as gauss
import pdb

class spectrogram:
  # Attributes
  # Public
  F_sample = 7000 # samle rate in Hz
  T_recording = None # duration of recording in seconds
  N = None # points in the DFT
  T_eval = None # Interval to evaluate DFT at in seconds
  # Private
  _waveform = None # Stored waveform
  _spectro = None # The spectrogram matrix
  _gnb = None # Gaussian Naive Bayes object

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
    #self._spectro = np.matrix(np.zeros((int(self.T_recording/self.T_eval), self.N)))
    self._spectro = np.matrix(np.zeros((int(np.shape(self._waveform)[0]*2/self.N), int(self.N/2))))
    # Evaluate the dft at timed intervals
    step = int(self.T_eval*self.F_sample)
    for i in range(int(np.shape(self._waveform)[0]*2/(self.N))):
    #for i in range(int(self.T_recording/self.T_eval)):
      #dft = np.fft.fft(self._waveform[i*step:i*step+self.N],axis=1)
      dft = np.fft.fft(self._waveform[i*int(self.N/2):i*int(self.N/2)+self.N],n=self.N*2,axis=0)
      dft = np.array([20 * np.log10(dft[i]) for i in range(np.shape(dft)[0])])
      # Normalize
      dft = dft/max(dft)
      if np.shape(dft)[0] != self.N*2:
        continue
      self._spectro[i,:] = dft.T[0,:int(self.N/2)]

  def plot(self):
    if self._spectro is None:
      print("ERROR: spectrogram has not been evaluated")
      return
    plt.figure(1)
    plt.subplot(141)
    plt.pcolormesh(self._spectro.T)
    plt.subplot(142)
    plt.plot(self._waveform)
    plt.subplot(143)
    spectrum, freqs, t, im = plt.specgram(np.reshape(self._waveform, (np.shape(self._waveform)[0],)), self.N) # look for docs on how this works
    plt.subplot(144)
    plt.plot(self._spectro[30,].T)
    print(np.shape(spectrum))
    print(np.shape(self._spectro))
    print(np.shape(self._waveform))
    plt.show()
  
  def save_spec(self, filename, start, stop):
    if self._spectro is None:
      print("ERROR: spectrogram has not been produced")
      return
    np.save(filename, self._spectro[start:stop,:])
    

  def create_gaussian(self, filenames):
    data = [np.load(f) for f in filenames]
    data_arr = data[0]
    for i in range(len(data)-1):
      data_arr = np.concatenate((data_arr,data[i+1]))

    classes = []
    curr_class = 0
    for d in data:
      classes = classes + [curr_class for i in range(np.shape(d)[0])]
      curr_class += 1
    #print(classes)
    #print(np.shape(data_arr))

    self._gnb = gauss()
    self._gnb.fit(data_arr, classes)
    #print(self._gnb.predict(data_arr[30,:].reshape(1,-1)))

  def predict(self):
    if self._spectro is None:
      print("ERROR: spectrogram has not been evaluated")
      return
    
    print(self._gnb.predict(np.asarray(self._spectro)))


def main():
  s = spectrogram(4, N=1024)
  s.record()
  s.eval()
  #s.plot()
  #start = int(input("First index: "))
  #stop = int(input("Last index: "))
  #filename = input("Filename: ")
  #s.save_spec(filename, start, stop)
  s.create_gaussian(("ah.npy", "ss.npy", "silence.npy"))
  s.predict()


if __name__ == "__main__":
  main()
