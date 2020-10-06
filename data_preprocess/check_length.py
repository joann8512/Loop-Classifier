import os
import numpy as np
import glob

def main():
    #files = glob.glob('/home/joann8512/NAS_189/home/LoopClassifier/npy_fixed/*')
    files = glob.glob('/home/joann8512/NAS_189/home/LoopClassifier/full_npy/*')
    for npy in files:
        length = len(np.load(npy))
        if length < 48000:
            os.remove(npy)
            
            
if __name__ == "__main__":
    main()
        
        
