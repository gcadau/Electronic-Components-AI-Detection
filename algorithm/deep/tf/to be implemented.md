- **Remove this file after implementation**.

Transfer learning and variations to be implemented

&rarr; 2 types (-> second type):


    1. Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset.
       You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.

    2. Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.


&rarr; Different possible pre-trainded weitghs.


&rarr; Implement: https://www.tensorflow.org/tutorials/images/transfer_learning?hl=en

&rarr; Implement: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub?hl=en
