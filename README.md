# Feature Augmentation based on Manifold Ranking and LSTM for Image Classification

**Authors:** Vanessa Helena Pereira-Ferrero, Lucas Pascotti Valem, Daniel Carlos Guimaraes Pedronette

**Overview**

This code is based on an LSTM network implementation using Python and Keras,
which initially used the MNIST dataset. For this framework purposes, it is necessary 
a file with the dataset classes (.txt file), the previous extraction of datasets features
(.npy file), and ranked lists using LHRR (Log-based Hypergraph of Ranking References)
manifold learning method (.txt file). Here the files were obtained thorough
UDLF - Unsupervised Distance Learning Framework. In this example, it is used Oxford
Flowers17 dataset classes, ResNet 152 features, and LHRR ranking files. 

**Research article**

The results are presented and compared in the paper entitled *"Feature Augmentation based 
on Manifold Ranking and LSTM for Image Classification"* published in 
*Expert System With Applications* journal, by academic publishing company *Elsevier*.

**DOI:**
https://doi.org/10.1016/j.eswa.2022.118995

**Abstract**

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

**Resources and repositories:**

LSTM for MNIST: https://github.com/ar-ms/lstm-mnist

UDLF framework: https://github.com/UDLF/UDLF

Oxford Flowers 17 dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html

LHRR manifold learning paper: https://ieeexplore.ieee.org/document/8733193


**Acknowledgments**

The authors are grateful to S??o Paulo Research Foundation - FAPESP, Brazil (grants #2020/02183-9, #2018/15597-6, #2020/11366-0 and #2017/25908-6), Brazilian National Council for Scientific and Technological Development- CNPq (grants #309439/2020-5, and #422667/2021-8) and Microsoft Research .

**S??o Paulo Research Foundation - FAPESP - related projects:**

grant #2020/02183-9

https://bv.fapesp.br/en/bolsas/191230/rank-based-unsupervised-learning-through-deep-learning-in-diverse-domains/

grant #2018/15597-6

https://bv.fapesp.br/en/auxilios/103794/aplication-and-investigation-of-unsupervised-learning-methods-in-retrieval-and-classification-tasks/

grant #2020/11366-0

https://bv.fapesp.br/en/bolsas/194260/support-for-computational-environments-and-experiments-execution-weakly-supervised-and-classificati/

grant #2017/25908-6

https://bv.fapesp.br/en/auxilios/102700/weakly-supervised-learning-for-compressed-video-analysis-on-retrieval-and-classification-tasks-for-v/

**Key-words:**

Image Classification, Feature Augmentation, LSTM, Manifold Learning, Ranking
