"""

Main implementation file

---------------------------------------------------------------------------------
Feature Augmentation based on Manifold Ranking and LSTM for Image Classification.
---------------------------------------------------------------------------------

@Authors and Contributors: 
    Vanessa Helena Pereira-Ferrero <nessahelena@gmail.com>
    Lucas Pascotti Valem <lucas.valem@unesp.br>
    Daniel Carlos Guimarães Pedronette <daniel.pedronette@unesp.br>

---------------------------------------------------------------------------------

This code is based on an LSTM network implementation using Python and Keras, 
which initially used the MNIST dataset. For this framework purposes, it is necessary a file with the dataset classes (.txt file), the previous extraction of datasets features 
(.npy file), and ranked lists using LHRR (Log-based Hypergraph of Ranking References)
manifold learning method (.txt file). Here the files were obtained thorough 
UDLF - Unsupervised Distance Learning Framework. In this example, it is used Oxford 
Flowers17 dataset classes, ResNet 152 features, and LHRR ranking files. The results are presented and compared in the paper entitled "Feature Augmentation based on Manifold Ranking and LSTM for Image Classification" accepted for publication in "Expert System With Applications" journal, by academic publishing company Elsevier.

Resources and repositories: 
    LSTM for MNIST: https://github.com/ar-ms/lstm-mnist
    UDLF framework: https://github.com/UDLF/UDLF
    Oxford Flowers 17 dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
    LHRR manifold learning paper: https://ieeexplore.ieee.org/document/8733193

---------------------------------------------------------------------------------

"""

# Imports
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


acum_exec_values = []

class LSTMClassifier(object):
    def __init__(self):
        # Classifier
        self.time_steps=28 # timesteps to unroll
        self.n_units=24 # hidden LSTM units
        self.n_inputs=256 # rows of pixels
        self.n_classes=17 # classes
        self.batch_size=128 # size of each batch
        self.n_epochs=10
        self.top_k= 100 #ranking positions
        self.components = 200 # PCA
        # Internal
        self._data_loaded = False
        self._trained = False
        # Test set size
        self.test_size = 200
        self.n_folds=10

    def __create_model(self):
        # LSTM Model creation and compilation
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(1, self.components)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        
    def read_ranked_lists_file(self, file_path):
        # Reading ranked list manifold file  
        print("\tReading file", file_path)
        with open(r"/data/flowers/_flowers17_CNN_ResNet_LHRR.txt", 'r') as f:
           return [[int(y) for y in x.strip().split(' ')][:self.top_k]
                   for x in f.readlines()]

    def load_data(self, fold_index):
        #Loading ranking and features
        self.rks = self.read_ranked_lists_file("/data/flowers/_flowers17_CNN_ResNet_LHRR.txt")
        self.features_original = np.load("/data/flowers/_flowers_features.npy") 
        print (self.features_original.shape)   
        self.features = []
        
        # Apply PCA on original features
        pca = PCA(n_components=self.components)
        self.features_original = pca.fit_transform(self.features_original)
  
        # load labels
        f = open("/data/flowers/_flowers_classes.txt")
        labels = [x.strip() for x in f.readlines()]
        f.close()
        labels = [int(x.split(":")[1]) for x in labels]
        # Split data in folds
        if self.folds is None:
            self.folds = self.fold_split(self.features_original, labels, n_folds= self.n_folds)
        train_index, test_index = self.folds[fold_index]
        self.train_index = train_index
        self.test_index = test_index
        self.labels = labels

        # Update features thorough rankings
  
        # load all the labels
        f = open("/data/flowers/_flowers_classes.txt")
        labels = [x.strip() for x in f.readlines()]
        f.close()
        labels = [x.split(":")[1] for x in labels]
        # compute top-k set average
        for i in range(len(self.features_original)):
            top_k_features = []
            if i in self.test_index:
              # test top-k
              for j in range(self.top_k): 
                  index = self.rks[i][j]
                  top_k_features.append(self.features_original[index])
            else:
              # train top-k
              j = 0
              while (len(top_k_features) != self.top_k) and (j < self.top_k):
                #print (len (top_k_features), str (j))
                index = self.rks[i][j]
                j += 1
                # skip if not train
                if i in self.test_index:
                  #print ("skip")  
                  continue
                class_elem = labels[index]
                #print (labels[index], labels[i])
                if class_elem == labels[i]:  # only add if the class is the same
                  top_k_features.append(self.features_original[index])
                
            # Set weights by alpha in first elements and p as an exponent
            # alpha = 0.6, 0.7, 0.8  p-value = 0.5, 0.6, 0.7    
            alpha = 0.6
            p = 0.7
            weights_array=sorted([(p**(x+1))*(1-alpha) for x in range(len(top_k_features))], reverse=True)
            weights_array [0] = alpha            
            new_feature = np.average(top_k_features, axis=0, weights=weights_array)
            
            self.features.append(new_feature)

       
        print(self.features[0])
        print(self.features_original[0])
        
        print(np.array_equal(self.features, self.features_original)) 
         
        
        
    def fold_split(self, features,
               labels,
               n_folds=10):
        # Split in folds
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        res = kf.split(features, labels)
        return list(res)  

          
    def train(self, fold_index, save_model=False):
        self.__create_model()
        self.load_data(fold_index)

        # train features
        x_train = []
        for index in self.train_index:
            x_train.append(self.features[index])
        print(len (x_train))
        x_train =  np.array(x_train).reshape(-1, 1, self.components)

        # labels features
        labels = self.labels
        labels_train = []
        for index in self.train_index:
            labels_train.append(labels[index])
        labels_train = to_categorical(labels_train)
        
        print(labels_train)
        
        self.model.fit(x_train, labels_train,
                  batch_size=self.batch_size, epochs=self.n_epochs, shuffle=True)

        self._trained = True
        
        if save_model:
            self.model.save("./saved_model/flowers-model.h5")

    def evaluate(self, fold_index, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.load_data(fold_index)

        x_test = []
        for index in self.train_index:
            x_test.append(self.features[index])
        x_test =  np.array(x_test).reshape(-1, 1, self.components)

        labels_test = []
        for index in self.train_index:
            labels_test.append(self.labels[index])
        labels_test = to_categorical(labels_test)

        
        print(labels_test)
        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, labels_test)
        acum_exec_values.append(test_loss)
        #f = open ("tmp_log.txt", "w+")
        print(test_loss)
        #f.close()


if __name__ == "__main__":
    lstm_classifier = LSTMClassifier()

    # gera os folds
    lstm_classifier.folds = None
    lstm_classifier.load_data(0)

    for fold_index in range(lstm_classifier.n_folds):
        print("Running for fold ", fold_index)
        lstm_classifier.train(fold_index, save_model=True)
        lstm_classifier.evaluate(fold_index)

    f = open ("tmp_log.txt", "w+")
    print(np.mean(acum_exec_values, axis=0).tolist(), file = f)
    print(acum_exec_values)
    f.close()  


