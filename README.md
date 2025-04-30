# Challenges of the Polimi course of Artificial Neural Networks and Deep Learning, 2024

Name of the group: *FORmidable*, composed by:

``` text
Kalisto Willaey
Matteo Pompilio
Tanguy Rolland
```

## Challenge 1: Image Classification on Blood Cells

In this first challenge, we face an image classification problem on a blood cells dataset. Our goal was to reach good accuracy on remote test set while achieving a great understanding of the methods to reach the best robustness.

The training set consists of circa 11000 samples of 10 classes of blood cells, with a resolution of 128x128 pixels, RGB:
![alt text](image-1.png)

We have used many data augmentation techniques and trained a variety of CNNs, while optimizing hyperparameters for a more efficient training.

Our final model is a MobileNetV3 non-pretrained with a custom head: we achieved a final accuracy of 80% on the hidden test set.
However, with further enhancement of the MobileNetV2 pretrained model, we could reach an accuracy of 86.97% on the hidden test set (notebook `FORmidable_MNV2_pretrained_best_hidden_set`), with the winning teams positioning at a 95% accuracy.

(you can find the full report on the pdf in `/HOM1/Report&Code/`)

## Challenge 2: Semantic Segmentation on Mars Surface

In this second challenge, we face an semantic segmentation problem on a mars surface dataset. Contrary to the first challenge, we put more focus on the models architectures and losses, since we already learnt how to implement a consistent augmentation pipeline from the former challenge.

The dataset consists of segmented 64x128 greyscale images from Mars terrain. Each image is paired with a mask representing the class of each pixel (Class Labels are: Background, Soil, Bedrock, Sand, Big Rock). Here is an example:
![alt text](image.png)

Since we couldn’t make use of pretrained models, hyperparameters tuning and training efficacy was paramount.

Our final model is a Enhanced (bottleneck) Double UNet: we achieved a final IoU (Intersection over Union) of 52.19% on the hidden test set.
However, with a further enhancement of the model (just by zero-weighting the background class), we could reach an IoU of 63.81% on the hidden test set (notebook `simple-double-0class0`), with the winning teams positioning at a 69% IoU.

(you can find the full report on the pdf in `/HOM2/Report&Code/`)
