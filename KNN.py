#------------------------
# Importing Liraries
#------------------------

import numpy as np
from sklearn.metrics import classification_report

#------------------------
# Getting Data
#------------------------

#url_or_filepath = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
#Datasets = np.genfromtxt(url_or_filepath, delimiter=",")

#------------------------
# Extracting Feature And Label
#------------------------

#Feature = Datasets[:,0:-1]
#Label   = Datasets[:,-1]

#------------------------
# calculating distance between data points
#------------------------

def EuclideanDistance(Train,Test_DataPoint):

    sum_arr, index = np.zeros((Train.shape[0]), dtype = np.float64), 0
    for Train_DataPoint in Train: 
        sum_ = 0
        for coordinate1,coordinate2 in zip(Test_DataPoint, Train_DataPoint):
            sum_ += (coordinate1 -coordinate2) ** 2
        sum_arr[index] = sum_ 
        index += 1 
    return np.sqrt(sum_arr)

#------------------------
# Distributing dataset into Train and Test set
#------------------------

def DistributeData(Feature, Label, Distribute_perc = 0.8, random_shuffle = False):

    if Distribute_perc < 1:
        Size_Train = int(Distribute_perc * Feature.shape[0])
    else:
        Size_Train = Distribute_perc

    if random_shuffle is True:
        rng_state = np.random.get_state()
        np.random.shuffle(Feature)
        np.random.set_state(rng_state)
        np.random.shuffle(Label)

    x_train = Feature[0:Size_Train,:]
    x_test  = Feature[Size_Train:,:]
    y_train = Label[0:Size_Train]
    y_test = Label[Size_Train:]

    return x_train, x_test, y_train, y_test

#------------------------
# Distributing Data
#------------------------

#x_train, x_test, y_train, y_test = DistributeData(Feature,Label,Distribute_perc = 0.75,random_shuffle = True)

#------------------------
# Getting k nearest Neighbor
#------------------------

def KNN_algorithm(x_train,y_train, x_test):
    Prediction_Array = np.zeros((x_test.shape[0],y_train.shape[0],2))
    Prediction_Array[:,:,0] = y_train
    index_ = 0

    for Test_DataPoint in x_test:
        Prediction_Array[index_,:,1] = EuclideanDistance(x_train ,Test_DataPoint) 
        index_ +=1
    return Prediction_Array

#------------------------
# Classifing Data
#------------------------

def classifier(k,x_train,y_train,x_test):
    Prediction_Array = KNN_algorithm(x_train,y_train, x_test)
    Prediction       = np.zeros(shape = Prediction_Array.shape[0])
    for i in range(Prediction_Array.shape[0]):
        d, max_ = dict(), 0
        for j in Prediction_Array[i][Prediction_Array[i,:,1].argsort()][-(k+1):-1,:][:,0]:
            if k>1:
                if j in d.keys():
                    d[j] += 1
                    if max_<d[j]:
                        max_ = d[j]
                        pred = j
                else:
                    d[j] = 1
                    pred = j
            else:
                pred =  Prediction_Array[i][Prediction_Array[i,:,1].argsort()][-(k+1):-1,:][:,0]
        Prediction[i] = pred
    return Prediction

#---------------------
# classification_report
#---------------------- 
def class_report(y_test,y_pred):
	print(classification_report(y_test,y_pred) )

#------------------------
# Getting Prediction
#------------------------

def predict(k, test):
    return classifier(k,test)


#------------------------
# Complete func
# -----------------------

def KNN_CLASSIFIER(url_or_filepath,k,test):

	Datasets = np.genfromtxt(url_or_filepath, delimiter=",")

	Feature = Datasets[:,0:-1]
	Label   = Datasets[:,-1]

	x_train, x_test, y_train, y_test = DistributeData(Feature,Label,Distribute_perc = 0.75,random_shuffle = True)

	y_pred = classifier(k,x_train,y_train,x_test)

	class_report(y_test,y_pred)
	test = np.genfromtxt(test, delimiter=",")
	return predict(k,test)
