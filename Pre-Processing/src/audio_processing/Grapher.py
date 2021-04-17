import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa as li
import librosa.display

class Grapher:
    def __init__(self, f):
        self.f = f

    def plot_waveforms(self, x, Fs):
        wave_fig, ax_wave = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(12,8))

        ax_wave[0, 0].plot(x, color='gray')
        ax_wave[0, 0].set_xlim([0, x.shape[0]])
        ax_wave[0, 0].set_xlabel(f'Time (in samples @ {Fs})')
        ax_wave[0, 0].set_ylabel('Amplitude')
        ax_wave[0, 0].set_title(f'Raw Wave: {self.f}')

        wave_fig.tight_layout()
        return wave_fig

    def plot_audiofeatures(self):
        pass

    def plot_tempos(self):
        pass

    def show_all(self):
        plt.show()

"""
stft_img = librosa.display.specshow(stft_db, y_axis='linear', x_axis='time', ax=ax_wave[0,1])
wave_fig.colorbar(stft_img, ax=ax_wave[0,1], format="%+2.f dB")
ax_wave[0,1].set_title(f'STFT: {f}')

mel_spec_db_img = librosa.display.specshow(mel_spectrogram_db, y_axis='mel', x_axis='time', ax=ax_wave[1,0])
wave_fig.colorbar(mel_spec_db_img, ax=ax_wave[1,0], format="%+2.f dB")
ax_wave[1,0].set_title(f'Mel Spectrogram (DB): {f}')

librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), #from magphase
    y_axis='log', x_axis='time', ax=ax_wave[1,1])
ax_wave[1,1].set_title(f'Log Power Spectrum: {f}')
spec_times = librosa.times_like(spec_centroid)
ax_wave[1,1].plot(spec_times, spec_centroid.T, label='Spectral centroid', color='w', alpha=1)
ax_wave[1,1].legend(loc='upper right')

wave_fig.tight_layout()

#Audio Feature Information
afeat_fig, ax_afeat = plt.subplots(nrows=2, sharex=False, figsize=(8,8))

mfccs_img = librosa.display.specshow(multi_mfccs_feature, y_axis='mel', x_axis='time', sr=Fs, ax=ax_afeat[0])
afeat_fig.colorbar(mfccs_img, ax=ax_afeat[0], format="%+2.f dB")
ax_afeat[0].set_title(f'MFCCs (39): {f}')
ax_afeat[0].set_xlabel(f'Time (in frame windows @ {n_fft} samples/window)')

chroma_img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax_afeat[1])
afeat_fig.colorbar(chroma_img, ax=ax_afeat[1], format="%+2.f dB")
ax_afeat[1].set_title(f'Chroma Pitch Class {f}')
afeat_fig.tight_layout()

#Tempo Information
tempo_fig, ax_temp = plt.subplots(nrows=3, sharex=True, figsize=(8,8))
times = librosa.times_like(oenv, sr=Fs)
ax_temp[0].plot(times, oenv, label='Onset strength')
ax_temp[0].legend(frameon=True)
ax_temp[0].label_outer()
ax_temp[0].set_title(f'Onset envelopes: {f}')
ax_temp[0].vlines(times[odetect], 0, oenv.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')
ax_temp[0].legend()

librosa.display.specshow(np.abs(tempogram), sr=Fs, hop_length=hop_length,
                         x_axis='time', y_axis='fourier_tempo', cmap='magma',
                         ax=ax_temp[1])
ax_temp[1].set_title(f'Fourier tempogram: {f}')
ax_temp[1].label_outer()
librosa.display.specshow(ac_tempogram, sr=Fs, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', cmap='magma',
                         ax=ax_temp[2])
ax_temp[2].set_title(f'Autocorrelation tempogram: {f}')
ax_temp[2].set_xlabel(f'Time (in frame windows @ {n_fft} samples/window)')

tempo_fig.tight_layout()

plt.show()
"""
