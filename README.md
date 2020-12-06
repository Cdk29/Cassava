# Cassava Cassava Leaf Disease Classification


Repository for the [Cassava Leaf Disease Classification Competition on Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification).
Data are 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. 

The task of the competition is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

# Organisation

## Learning rate finder for the efficientnet with R

In the notebook [Cassava_lr_finder.md](https://github.com/Cdk29/Cassava/blob/main/Cassava_lr_finder.md) first step in the model's training workflow is to determine
the range of the (cyclical) learning rate to use to train the efficientnet. It is done in this notebook. The output of the learning rate finder looks like this :

![LR_Finder](https://github.com/Cdk29/Cassava/blob/main/Cassava_lr_finder_files/figure-gfm/unnamed-chunk-32-1.png)

## Training and test a model using the application wrapper of efficientnet

The notebook [cassava_application_efficientnet.md](https://github.com/Cdk29/Cassava/blob/main/cassava_application_efficientnet.md) is oldest than the previous one, since I needed first to test if the application wrapper for the efficientnets in this 
[forked version of keras](https://github.com/Cdk29/keras). This notebook use cyclical learning rate to train a model create using the keras sequential API,
using wrapper of efficinetnet that are not yet deployed in the main distribution of keras, but has been first implemented 
[here](https://github.com/rstudio/keras/commit/c406ec55f7bb2864ac58a17f963448810a531c18). The details of the implementation 
of the cyclical learning rate are inside the notebook.

![Training](https://github.com/Cdk29/Cassava/blob/main/cassava_application_efficientnet_files/figure-gfm/unnamed-chunk-36-1.png)

![Learning_rate](https://github.com/Cdk29/Cassava/blob/main/cassava_application_efficientnet_files/figure-gfm/unnamed-chunk-30-2.png)

## Fine tuning of the Efficientnet-B0

In the notebook [cassava_file_tuning.md](https://github.com/Cdk29/Cassava/blob/main/cassava_file_tuning.md) I fine tuned the EfficientNet-B0, from (currently) the arbitrary layer of block5a_expand_conv. 

![Fine_tuning](https://github.com/Cdk29/Cassava/blob/main/cassava_file_tuning_files/figure-gfm/unnamed-chunk-39-1.png)
