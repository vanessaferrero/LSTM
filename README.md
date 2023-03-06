# LSTM
Feature Augmentation based on Manifold Ranking and LSTM for Image Classification
Vanessa Helena Pereira-Ferrero, Lucas Pascotti Valem, Daniel Carlos Guimaraes Pedronette
This code is based on an LSTM network implementation using Python and Keras,
which initially used the MNIST dataset. For this framework purposes, it is necessary 
a file with the dataset classes (.txt file), the previous extraction of datasets features
(.npy file), and ranked lists using LHRR (Log-based Hypergraph of Ranking References)
manifold learning method (.txt file). Here the files were obtained thorough
UDLF - Unsupervised Distance Learning Framework. In this example, it is used Oxford
Flowers17 dataset classes, ResNet 152 features, and LHRR ranking files. 
The results are presented and compared in the paper entitled "Feature Augmentation based 
on Manifold Ranking and LSTM for Image Classification" accepted for publication in 
"Expert System With Applications" journal, by academic publishing company Elsevier.
Image classification is a critical topic due to its wide application and associated challenges. 
Despite the considerable progress made last decades, there is still a demand for
context-aware image representation approaches capable of taking into the dataset
manifold for improving classification accuracy. In this work, a representation learning
approach is proposed, based on a novel feature augmentation strategy. The proposed
method aims to exploit available contextual similarity information through rank-based
manifold learning is used to define and assign weights to samples used in
augmentation. The approach is validated using CNN-based features and LSTM models
to achieve even higher accuracy results on image classification tasks. Experimental
results show that the feature augmentation strategy can indeed improve the accuracy
of results on widely used image datasets.

#Resources and repositories:
#LSTM for MNIST: https://github.com/ar-ms/lstm-mnist
#UDLF framework: https://github.com/UDLF/UDLF
#Oxford Flowers 17 dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
#LHRR manifold learning paper: https://ieeexplore.ieee.org/document/8733193

Image Classification

Feature Augmentation

LSTM

Manifold Learning

Ranking
