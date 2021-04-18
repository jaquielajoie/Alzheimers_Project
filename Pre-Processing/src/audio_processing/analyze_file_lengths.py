import os
from file_lengths import FileLengths
import pandas as pd
import numpy as np
import json

#path = os.path.abspath('../file_lengths.json')
fl = FileLengths()

df = np.array(fl.file_lengths)
#file_lengths = json.loads(path)
df = np.delete(df, 1, axis=1)
df = np.squeeze(df)
df = df.astype(np.float)

#35 seconds as a cutoff
hist, bin_edges = np.histogram(df, bins=20, range=(0,40))
print(hist)
print(bin_edges)


with open(os.path.abspath('json/data.json'), "r") as fp:
    data = json.load(fp)

dementia = 0
control = 0
for i in data["labels"]:
    if i == 0:
        dementia += 1
    if i == 1:
        control += 1

print(f'Dementia samples: {dementia}\nControl samples: {control}')

"""
m = 0
mn = 5000
sm = 0

for f in fl.file_lengths:
    m = max(m, f[0])
    mn = min(mn, f[0])
    sm += f[0]
    print(f[0])

print(f'MAX: {m}\nMIN: {mn}\nAVG: {sm/len(fl.file_lengths)}')

"""
