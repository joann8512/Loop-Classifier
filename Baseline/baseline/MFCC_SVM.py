import numpy as np
from sklearn.svm import SVC
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance
import librosa
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from skmultilearn.ensemble import RakelD
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import json
import pandas as pd



def main():
    tr = np.load('/home/joann8512/NAS_189/home/LoopClassifier/train_full.npy')  # all names are unique, all binary labels
    te = np.load('/home/joann8512/NAS_189/home/LoopClassifier/test_full.npy')
    val = np.load('/home/joann8512/NAS_189/home/LoopClassifier/valid_full.npy')
    used = []
    for i, fn in enumerate(tr):
        used.append(fn.split('_')[0].split('\t')[-1])
    for i, fn in enumerate(te):
        used.append(fn.split('_')[0].split('\t')[-1])
    for i, fn in enumerate(val):
        used.append(fn.split('_')[0].split('\t')[-1])
    
    path = '/home/joann8512/NAS_189/home/LoopClassifier/features/'  # all json files
    #temp = '/home/joann8512/NAS_189/home/LoopClassifier/annotations/**/*.json'
    extract = os.listdir(path)
    #annot = glob.glob(temp)
    #annot = [i.split('-')[-1] for i in annot]
    all_data = []  # all path to feature data
    for ids in extract:
        if ids.split('.')[0] in used:
            all_data.append(os.path.join(path, ids))
        
    idx_list = []
    df = pd.read_csv('/home/joann8512/NAS_189/home/LoopClassifier/new_full.csv')
    ids = [i.split('_')[0] for i in df['id']]
    for data in all_data:
        idx_list.append(ids.index(data.split('/')[-1].split('.')[0]))
    binary = []  
    fpath = []
    for i in range(len(idx_list)):
        fpath.append(list(df.iloc[idx_list[i]])[0])
        binary.append(list(df.iloc[idx_list[i]][1:]))
        
    all_features = []
    fpath = [j.split('_')[0] for j in fpath]
    all_data_fn = [i.split('/')[-1].split('.')[0] for i in all_data]

    use_idx = []
    for k in fpath:
        use_idx.append(all_data_fn.index(k))

    for idxs in use_idx:
        #print(all_data[idxs])
        with open(all_data[idxs]) as f:
            data = json.load(f)
        feature = [p for q in data['analysis']['lowlevel']['mfcc'].values() for p in q]
        all_features.append(feature)    
        
        
    cut = int(len(all_features)*0.9)
    train_bin = np.array(binary[:cut])
    test_bin = np.array(binary[cut:])

    train_feat = np.array(all_features[:cut])
    test_feat = np.array(all_features[cut:])


    print("### BR-SVM ###")
    clf = BinaryRelevance(
        classifier = SVC(kernel='linear', C=100),
        require_dense = [False, True]
    )
    clf.fit(train_feat, train_bin)
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))
    

    print("### BR-RF ###")
    clf = BinaryRelevance(
        classifier = RandomForestClassifier(n_estimators=1000),
        require_dense = [False, True]
    )
    clf.fit(train_feat, train_bin)
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))
    #fpr, tpr, thresholds = roc_curve(test_bin, prediction.toarray())
    #print('AUC score is {}'.format(metrics.auc(fpr, tpr)))
    

    print("### LP SVM")
    clf = LabelPowerset(
        classifier = SVC(kernel='linear', C=10),
        require_dense = [False, True]
    )
    
    # train
    clf.fit(train_feat, train_bin)
    
    # predict
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))
    
    print("### LP RF")
    clf = LabelPowerset(
    classifier = RandomForestClassifier(n_estimators=1000),
    require_dense = [False, True]
    )

    # train
    clf.fit(train_feat, train_bin)

    # predict
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))
   
    print("### RAKEL SVM ###")
    clf = RakelD(
    base_classifier=SVC(kernel='linear', C=1),
    base_classifier_require_dense=[False, True],
    labelset_size=3
    )

    # train
    clf.fit(train_feat, train_bin)

    # predict
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))
    
    print("### RAKEL RF ###")
    clf = RakelD(
    base_classifier=RandomForestClassifier(n_estimators=1000),
    base_classifier_require_dense=[False, True],
    labelset_size=3
    )

    # train
    clf.fit(train_feat, train_bin)

    # predict
    prediction = clf.predict(test_feat)
    #print('Test accuracy is {}'.format(accuracy_score(test_bin, prediction)))
    #print('Hamming loss is {}'.format(hamming_loss(test_bin, prediction)))
    print('F1 score is {}'.format(f1_score(test_bin, prediction.toarray(), average='samples')))
    print('Average precision is {}'.format(average_precision_score(test_bin, prediction.toarray())))
    print('roc_auc_score is {}'.format(roc_auc_score(test_bin, prediction.toarray())))


if __name__ == "__main__":
    main()
    
