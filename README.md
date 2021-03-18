# Loop-Classifier

The proposed work introduces a new type of auto-taggingtask, called instrument role classification. We  discuss why the task is necessary, especially under the setting of making electronic music. Loopbased music and the general background regarding the creation of thisspecific style is also introduced. We approach this task with a previouslyintroduced  Convolutional Neural  Network  architecture,  the  HarmonicCNN  (HCNN). A  new  use  case  of  this  method  is  presented  by  usingthe Freesound Loop Dataset (FSLD), emphasizing its learning efficiencyunder limited data. Finally, we present baselines to highlight these ad-vantages.

## FreeSound Loop Dataset (FSLD)
To download the dataset used in this work, please refer to [FSLD](https://zenodo.org/record/3967852) for the data.
And please cite this paper if you use this dataset:
```ruby
@inproceedings{ramires2020, author = "Antonio Ramires and Frederic Font and Dmitry Bogdanov and Jordan B. L. Smith and Yi-Hsuan Yang and Joann Ching and Bo-Yu Chen and Yueh-Kao Wu and Hsu Wei-Han and Xavier Serra", title = "The Freesound Loop Dataset and Annotation Tool", booktitle = "Proc. of the 21st International Society for Music Information Retrieval (ISMIR)", year = "2020" }
```

## Pretrained model
To use the pretrained model of this work, set path to data and run ```ruby loop_eval.py```

## Cite
[1] Joann Ching, Antonio Ramires, Yi-Hsuan Yang. "Instrument Role Classification: Auto-tagging for Loop Based Music" (KTH'20)