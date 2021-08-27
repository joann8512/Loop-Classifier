# Loop-Classifier

The proposed work introduces a new type of auto-taggingtask, called instrument role classification. We  discuss why the task is necessary, especially under the setting of making electronic music. Loopbased music and the general background regarding the creation of thisspecific style is also introduced. We approach this task with a previouslyintroduced  Convolutional Neural  Network  architecture,  the  HarmonicCNN  (HCNN). A  new  use  case  of  this  method  is  presented  by  usingthe Freesound Loop Dataset (FSLD), emphasizing its learning efficiencyunder limited data. Finally, we present baselines to highlight these ad-vantages.

## FreeSound Loop Dataset (FSLD)
To obtain the dataset used in this work, please refer to [FSLD](https://zenodo.org/record/3967852) for the data.  
And please cite this paper if you use this dataset:
```ruby
@inproceedings{ramires2020, author = "Antonio Ramires and Frederic Font and Dmitry Bogdanov and Jordan B. L. Smith and Yi-Hsuan Yang and Joann Ching and Bo-Yu Chen and Yueh-Kao Wu and Hsu Wei-Han and Xavier Serra", title = "The Freesound Loop Dataset and Annotation Tool", booktitle = "Proc. of the 21st International Society for Music Information Retrieval (ISMIR)", year = "2020" }
```

## Inferencing With Pretrained Model 
To use the pretrained model of this work, set path to data and run ```loop_eval.py```

## To Train the Model With Your Own Data
1. Run ```data_preprocess/data.py``` to get your audio data for training in npy format.
2. Run ```data_preprocess/split.py``` to split all data into train/valid/test.
3. Run ```data_preprocess/check_length.py``` to make sure the segments are all in the 3 seconds frame.
4. In ```loop_main.py``` , make sure to change the data path to the directory of your npy files, and you can start training!
5. Lastly, run ```loop_eval.py``` to check out the classification results.

## Cite
[1] Joann Ching, Antonio Ramires, Yi-Hsuan Yang. "Instrument Role Classification: Auto-tagging for Loop Based Music" (KTH'20)
