import librosa as li
import numpy as np
import math

class FeatureExtractor:
    def __init__(self, x, Fs, n_fft, hop_length, num_mfcc, num_segments=1):
        self.x = x
        self.Fs = Fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mfcc = num_mfcc
        self.num_segments = num_segments
        duration = li.get_duration(y=x, sr=Fs, n_fft=n_fft, hop_length=hop_length, center=True)
        SAMPLES_PER_TRACK = Fs * duration
        self.samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

        print(f'SAMPLES_PER_TRACK: {SAMPLES_PER_TRACK}\n# hops per track {SAMPLES_PER_TRACK/hop_length}')

    def get_mfccs(self):

        
        mfcc =  li.feature.mfcc(self.x, self.Fs, n_mfcc=self.num_mfcc, n_fft=self.n_fft,
                                     hop_length=self.hop_length)
        return mfcc.T.tolist()
        """
        mfccs = []
        num_mfcc_vectors_per_segment = math.ceil(self.samples_per_segment / self.hop_length)

        for seg in range(self.num_segments):
            print(seg)
            start = self.samples_per_segment * seg
            finish = start + self.samples_per_segment
            mfcc = li.feature.mfcc(self.x[start:finish], self.Fs, n_mfcc=self.num_mfcc, n_fft=self.n_fft,
                                         hop_length=self.hop_length)
            mfcc = mfcc.T
            print(mfcc)
            print(mfcc.shape)
            print(f'{len(mfcc)}\n\n')

             # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                print(True)
                mfccs.append(mfcc.tolist())
        print(f'full: {len(mfccs)}')
        return mfccs #np.array(mfccs)
        """



    """
    #TODO
    def describe_freq(self, freqs):
        mean = np.mean(freqs)
        std = np.std(freqs)
        maxv = np.amax(freqs)
        minv = np.amin(freqs)
        median = np.median(freqs)
        #skew = scipy.stats.skew(freqs)
        #kurt = scipy.stats.kurtosis(freqs)
        q1 = np.quantile(freqs, 0.25)
        q3 = np.quantile(freqs, 0.75)
        #mode = scipy.stats.mode(freqs)[0][0]
        #iqr = scipy.stats.iqr(freqs)

        return [mean, std, maxv, minv, median, q1, q3]
    """




















    def donotuse(self):
        return
        mfccs = librosa.feature.mfcc(x, Fs, n_mfcc=num_mfcc, n_fft=n_fft,
                                     hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs) #first derivative
        delta2_mfccs = librosa.feature.delta(mfccs, order=2) #second derivative
        multi_mfccs_feature = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

        chroma = librosa.feature.chroma_cqt(y=x, sr=Fs)

        stft = librosa.stft(x)  # STFT of x
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        S, phase = librosa.magphase(stft)
        spec_centroid = librosa.feature.spectral_centroid(S=S)

        mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=Fs)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        #Tempo information
        oenv = librosa.onset.onset_strength(y=x, sr=Fs, hop_length=hop_length)
        odetect = librosa.onset.onset_detect(y=x, sr=Fs, onset_envelope=oenv, hop_length=hop_length)
        tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=Fs,
                                                      hop_length=hop_length)
        # Compute the auto-correlation tempogram, unnormalized to make comparison easier
        ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=Fs,
                                                 hop_length=hop_length, norm=None)

        zero_crossings = librosa.feature.zero_crossing_rate(y=x,frame_length=n_fft,hop_length=hop_length,center=True)

        #spec_centroid = librosa.feature.spectral_centroid(y=x, sr=Fs, n_fft=n_fft, hop_length=hop_length, center=True)

        #TODO: need to break this down by frame
        #   Right now it describes the entire file.
        #   Useful for gender recognition
        #freqs = np.fft.fftfreq(x.size)
        #freq_description = self.describe_freq(freqs)

        print(f"""\nfile_path: {file_path}\n\nAudio Feature Shapes ( # features, # of frames sized {n_fft} samples):\n
        mfccs: {mfccs.shape}
        multi_mfccs: {multi_mfccs_feature.shape}
        chroma: {chroma.shape}
        stft_db: {stft_db.shape}
        mel_spectrogram_db: {mel_spectrogram_db.shape}
        fourier_tempogram: {tempogram.shape}
        onset_strength: {oenv.shape} ----> (1D list of time values)
        onset_detect: {odetect.shape} ----> (1D list of time values)
        tempogram: {tempogram.shape}
        ac_tempogram: {ac_tempogram.shape}
        zero_crossings: {zero_crossings.shape}
        spec_centroid: {spec_centroid.shape}
        """)
