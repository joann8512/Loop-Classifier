import os
import numpy as np
import pandas as pd
import glob
from essentia.standard import MonoLoader  # outputs mono audio signal
import fire
import tqdm

def get_npy(fn):
    fs = 16000
    loader = MonoLoader(filename=fn, sampleRate=fs)
    x = loader()
    return x

def remove_other():
    csv_df = pd.read_csv('/home/joann8512/NAS_189/home/LoopClassifier/audio_label2_fixed.csv')
    lst = []
    for line in range(len(csv_df)):
        if np.array(csv_df.loc[line][-1]) == 1:
            lst.append(line)
    csv_df3 = csv_df.drop(lst, axis=0)
    csv_df3.to_csv('/home/joann8512/NAS_189/home/LoopClassifier/audio_label_noOther.csv', index=False)
    

def main():
    fs = 16000
    data_path = "/home/joann8512/NAS_189/home/LoopClassifier/"
    #files = glob.glob(os.path.join(data_path, 'For_Instrument_Classification', 'audio', 'wav', '*.wav'))
    files = glob.glob(os.path.join(data_path, 'FSL10K', 'audio', 'wav', '*.wav'))
    npy_path = os.path.join(data_path, 'full_npy')  
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    used = []
    f = open(os.path.join(data_path, "full_used.txt"), "r")
    for x in f:
         used.append(x.strip('\n'))

    for fn in tqdm.tqdm(files):
        name = fn.split('/')[-1]  # fn = full wav file path
        #name = name.split('_')[0]
        #if name in open('/home/joann8512/NAS_189/home/LoopClassifier/audio_label2_fixed.csv').read():
        if name in used:
            npy_fn = os.path.join(npy_path, fn.split('/')[-1].split('.')[0]+'.npy')
            if not os.path.exists(npy_fn):
                try:
                    x = get_npy(fn)
                    np.save(open(npy_fn, 'wb'), x)
                except RuntimeError:
                    # some audio files are broken
                    print(fn)
                    continue
                
                
if __name__ == ("__main__"):
    main()
