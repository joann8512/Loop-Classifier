import os
import numpy as np
import pandas as pd

def main():
    save_path = "/home/joann8512/NAS_189/home/LoopClassifier/"
    #csv_df = pd.read_csv(os.path.join(save_path, 'audio_label2_fixed.csv'))
    csv_df = pd.read_csv(os.path.join(save_path, 'new_full.csv'))
    binary = np.zeros((2936, 6))  # 1856
    titles = []
    #idx = 0
    for i in range(0, 2936):
        features = np.array(csv_df.loc[i][1:])
        title = csv_df.loc[i][0]
        #if np.sum(features) != 0:
        binary[i] = features
        #idx += 1
        titles.append(title)
    
    #drop_list = []
    #for i in range(len(titles)-1):
    #    for j in range(i+1, len(titles)):
    #        if titles[j] == titles[i]:
    #            if j not in drop_list:
    #                drop_list.append(j)
    
    #filenames = []
    #for i in range(len(titles)):
    #    if i not in drop_list:
    #        filenames.append(titles[i])
    
    npy_list = os.listdir('/home/joann8512/NAS_189/home/LoopClassifier/full_npy')
    npy_list = [i.split('.')[0] for i in npy_list]
    
    filenames = titles        
    drop = []  # list of files less than 3 sec      
    for name in filenames:
        if name.split('.')[0] not in npy_list:
            #drop.append(name)
            drop.append(filenames.index(name))
    #for d in drop:
    #    filenames.remove(d)
    short_binary = []
    short_filenames = []
    for j in range(len(filenames)):
        if j not in drop:
            short_binary.append(binary[j])
            short_filenames.append(filenames[j])
    short_binary = np.array(short_binary)        
            
    #binary = np.delete(binary, drop, axis = 0)
    np.save(open(os.path.join(save_path, 'binary_full.npy'), 'wb'), short_binary)

    tr = []
    val = []
    test = []
    for i, title in enumerate(short_filenames):  # titles -> filenames -> short_filenames
        if i < len(short_filenames)*0.9:
            if short_binary[i].sum() > 0:
                tr.append(str(i)+'\t'+title)
        elif len(short_filenames)*0.9 <= i < len(short_filenames)*0.95:
            if short_binary[i].sum() > 0:
                val.append(str(i)+'\t'+title)
        else:
            if short_binary[i].sum() > 0:
                test.append(str(i)+'\t'+title)
    np.save(open(os.path.join(save_path, 'train_full.npy'), 'wb'), tr) #tr_exist)
    np.save(open(os.path.join(save_path, 'valid_full.npy'), 'wb'), val) #val_exist)
    np.save(open(os.path.join(save_path, 'test_full.npy'), 'wb'), test)
    
    
if __name__ == "__main__":
    main()
    
    