import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def display_waveform(y, sr, name):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12,3))
    plt.plot(times, y, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{name} Waveform")
    plt.tight_layout()
    plt.show()
    plt.close()
    
def display_stft(y, sr, name):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{name} STFT")
    plt.tight_layout()
    plt.show()
    plt.close()
    
def display_frequency_graph(y, sr, name):
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    magnitude_db = librosa.amplitude_to_db(np.abs(fft), ref=np.max)
    plt.figure(figsize=(12, 4))
    plt.semilogx(freqs[1:], magnitude_db[1:], linewidth=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{name} Frequency Spectrum")
    plt.xlim(20, sr / 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def display_mel(y, sr, name):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{name} Mel Spectrogram")
    plt.tight_layout()
    plt.show()
    plt.close()