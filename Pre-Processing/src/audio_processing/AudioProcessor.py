import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import re

import librosa
import librosa.display
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import audioread as ar

import os
import json
import sys
import threading
import time
import warnings
import math

from .Grapher import Grapher
from .FeatureExtractor import FeatureExtractor
from .TrainingLabeler import TrainingLabeler

threadLock = threading.Lock()


"""
*
* Handles file conversion, OS Walk
* Outer interface for feature analysis and extraction
*
"""
class AudioProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.mp3s = f'{data_path}/mp3'
        self.uncompressed_formats = ['.wav', '.WAV', '.pcm', '.PCM']
        self.file_lengths = []

    """
    File Conversion Start
    """
    def enforce_mp3(self, file_path, extension):
        if extension in ['.mp3', '.MP3']: return #no need.
        mp3_filename = os.path.splitext(os.path.basename(file_path))[0] + '.mp3'
        seg = AudioSegment.from_file(file_path)
        seg = seg.set_frame_rate(22050)
        seg.export(f'../test_data/mp3/{mp3_filename}', format='mp3')

        print(f'\nGenerated {mp3_filename} and exported to ../test_data/mp3/')
        return

    def standardize_audio_files(self):
        print(f'\nbeginning search at {self.data_path}\n')
        start_time = time.perf_counter()
        file_count = 0
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.data_path)):
            # if dirpath not in self.ignore_dirs:
            # save sub-folder name in the mapping
            label = dirpath.split("/")[-1]
            print(f'Searching {label} directory...')

            for f in filenames:
                filename, extension = os.path.splitext(f)
                if extension in self.uncompressed_formats:
                    file_path = os.path.join(dirpath, f)
                    self.enforce_mp3(file_path=file_path, extension=extension)
                    file_count += 1
        end_time = time.perf_counter()
        print(f'\nfunc: standardize_audio_files took {end_time - start_time} seconds to run.\nAnalyzed {file_count} files.')
    """
    File Conversion End
    """

    ###############################################################

    """
    *
    * Analyze MP3s and export MFCCs & labeling to ./json/data.json
    *
    """
    def add_file_features_to_labeler(self, training_labeler, x, Fs, mms, file_path, i, f):

        file_dur = librosa.get_duration(y=x, sr=Fs)

        print(f'\n============\nAudio Length: {librosa.get_duration(y=x, sr=Fs)}\n============\n')
        self.file_lengths.append((librosa.get_duration(y=x, sr=Fs), str(file_path)))

        n_mfcc = 13
        n_fft = 2048
        hop_length = 512



        duration = 35 #seconds, all files shorter are dropped
        SAMPLES_PER_SEGMENT = Fs * duration

        segments_per_file = math.floor(file_dur/duration)

        #num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_num_mfcc_vec_p_seg = math.ceil(SAMPLES_PER_SEGMENT / hop_length)



        for s in range(segments_per_file):
            start_sample = SAMPLES_PER_SEGMENT * s #s=0 -> 0
            finish_sample = start_sample + SAMPLES_PER_SEGMENT #s=0 -> num_samples_per_segment

            print(f'Start sample: {start_sample}\nFinish sample: {finish_sample}')

            #if len(x[start_sample:finish_sample]) == base_sample_count * sample_factor: #30 seconds #1984500:#396900: #need the same shape...

            mfcc = librosa.feature.mfcc(
                                    x[start_sample:finish_sample],
                                    sr=Fs,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length
                                )

            chroma_stft = librosa.feature.chroma_stft(x[start_sample:finish_sample],
                                    sr=Fs,
                                    n_chroma=12,
                                    n_fft=n_fft,
                                    hop_length=hop_length
                                )

            mfcc = mfcc.T
            chroma_stft = chroma_stft.T

            feat = np.concatenate((mfcc, chroma_stft), axis=1)
            #print(f'\nSample Factor: {sample_factor}')
            print(f'\n\nFs: {Fs}\nfeat.shape: {feat.shape}\nlen(feat): {len(feat)}\n expected_num_mfcc_vec_p_seg: {expected_num_mfcc_vec_p_seg}')
            #print(f'{mfcc.shape}\n{chroma_stft.shape}')

            #print(mfcc)

            if len(feat) == expected_num_mfcc_vec_p_seg:
                feat = feat.tolist()

                print(f'\nACCEPTED\n[LABEL={i-1}, MMS={mms[1]}]\n\n----------------\n')
                training_labeler.add_information(label=(i-1), feat=feat, file_name=f, extra_info=None, mms=mms)

    def get_features_from_file_path(self, training_labeler, file_path, filename, i, f, mms_df=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, Fs = librosa.load(file_path, sr=16000)

        #extra_info = ex_df[ex_df['id']==file_num]
        if mms_df is not None:
            fn_reg = re.compile('([0-9]+)')
            file_num = int(fn_reg.findall(filename)[0])

            mms = mms_df[mms_df['id']==file_num]
            mms = np.squeeze(mms)
            mms = mms.values.tolist()
        else:
            mms = [-1, -1] #for training labeler

        self.add_file_features_to_labeler(training_labeler=training_labeler, x=x, Fs=Fs, mms=mms, file_path=file_path, i=i, f=f)

    def parse_metadata(self, run=True):
        if run:
            pitt_df = pd.read_csv(os.path.abspath("../test_data/Pitt-data.csv"))
            cols = ["id","entryage","onsetage","sex"]
            mms_cols = ["id","mms"]
            ex_df = pitt_df[cols]
            mms_df = pitt_df[mms_cols]
            return mms_df, ex_df
        else:
            return None, None

    def analyze_mp3s(self, cmd_print=False, max_file_count=None, play_audio=False, audio_length=5, parse_metadata=True):
        start_time = time.perf_counter()
        file_count = 0
        features_per_file = 0

        mms_df, ex_df = self.parse_metadata(run=parse_metadata)

        training_labeler = TrainingLabeler(os.path.abspath("audio_processing/json/data.json"))

        try:

            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.mp3s)):
                label = dirpath.split("/")[-1]
                training_labeler.add_information(mapping=label)
                print(f'\nPulling from {label} directory...')
                for f in filenames[:max_file_count]:
                    filename, extension = os.path.splitext(f)
                    if extension in ['.mp3', '.MP3']:
                        features_per_file = 0 #just gets the last file analyzed, should be equal for all.
                        file_count += 1
                        file_path = os.path.join(dirpath, f)

                        self.get_features_from_file_path(training_labeler=training_labeler, file_path=file_path, filename=filename, mms_df=mms_df, i=i, f=f)

                        #Librosa throws an audioread UserWarning - supressed below
                        if cmd_print:
                            if play_audio:
                                ap = AudioPlayer(threadId=1, name=f, file_path=file_path, audio_length=audio_length)
                                ap.start()

                            grapher = Grapher(f=f)
                            #Waveform Information
                            grapher.plot_waveforms(x=x, Fs=Fs)

                            grapher.show_all()

                            if play_audio:
                                ap.join()

            end_time = time.perf_counter()

            print(f'\nfunc: analyze_mp3s took {end_time - start_time} seconds to run.')
            print(f'{file_count} mp3 files analyzed.')
            print(f'{features_per_file} audio features per file (per frame).')
            print(f'Total number of entries: {training_labeler.total_entries}')

            training_labeler.save()

            #with open('file_lengths.json', "w") as fp:
                #json.dump(self.file_lengths, fp, indent=4)

        except Exception as e:
            print(e)
            training_labeler.save()


    def run(self, cmd_print=False, max_file_count=None, process_large_files=False, analyze_mp3s=True, play_audio=False, audio_length=5):
        if process_large_files: self.standardize_audio_files()
        if analyze_mp3s: self.analyze_mp3s(cmd_print=cmd_print, max_file_count=max_file_count, play_audio=play_audio, audio_length=audio_length)



"""
*
* Audio Player: plays files from command line during processing
* Uses lock at the top of the file
*
"""
class AudioPlayer(threading.Thread):
    def __init__(self, threadId, name, file_path, audio_length):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.file_path = file_path
        self.play_len = audio_length * 1000 # in seconds

    def run(self):
        print(f'\nStarting {self.name} audio player thread.')
        threadLock.acquire()
        aseg = AudioSegment.from_mp3(self.file_path)
        play(aseg[:self.play_len])
        threadLock.release()
        print(f'\nFinished playing {self.name}.')
        return
