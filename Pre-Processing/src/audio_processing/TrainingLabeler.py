import os
import json
import numpy as np
import re

class TrainingLabeler:
    def __init__(self, save_path):
        self.total_entries = 0
        self.data = {
            "mapping": [],
            "mms": [],
            "labels": [], #clarify usage: currently capturing folder depth
            "features": [],
            "files": [],
            "extra_info": [],

        }
        self.save_path = save_path

    """
    Mel-frequency Cepstral Coefficients (MFCCs), Chromagram, Mel-scaled Spectrogram, Spectral Contrast and Tonal Centroid Features (tonnetz).
    ADD MSSE score as well.
    """

    def add_information(self, mapping=None, label=None, feat=None, file_name=None, extra_info=None, mms=None):
        if mapping:
            self.data["mapping"].append(mapping)
        if label != None:
            self.data["labels"].append(label)
        if feat:
            self.data["features"].append(feat)
            self.total_entries += 1
        if file_name:
            self.data["files"].append(file_name)
            print(f'Extracting at {file_name}...')
        if mms:
            self.data["mms"].append(mms[1])
        if extra_info:
            print(f'\nextra_info: {extra_info}\ncols = ["id","entryage","onsetage","sex"]\n')
            self.data["extra_info"].append(extra_info[1:])

    def save(self):
        with open(self.save_path, "w") as fp:
            json.dump(self.data, fp, indent=4)
