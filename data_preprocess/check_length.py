import os
import numpy as np
import glob

def main():
    files = glob.glob('./LoopClassifier/full_npy/*')
    for npy in files:
        length = len(np.load(npy))
        if length < 48000:
            os.remove(npy)
            
            
if __name__ == "__main__":
    main()
        
        
