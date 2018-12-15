
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#Loading the data
M = loadmat('MNIST_digit_data.mat')

images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
print (images_train.shape)
print (labels_train.shape)

#**************** dataset pre-processing for question (a) and (b) ************
def filterDataset(num1, num2):
    #filter train dataset to contain only images for digit num1 & num2
    images_train_new, labels_train_new = [], [] 
    #take threshold samples for digit num1 and digit num2 each
    threshold = 1000
    no1, no2 = 0, 0
    for i in range(len(labels_train)):
        if no1 < threshold and labels_train[i][0] == num1:
            images_train_new.append(images_train[i])
    #        print (labels_train[i][0])
            labels_train_new.append([1])
            no1 += 1
        elif no2 < threshold and labels_train[i][0] == num2:
            images_train_new.append(images_train[i])
    #        print (labels_train[i][0])
            labels_train_new.append([-1])
            no2 += 1
    
    #filter test dataset to contain only images for digit num1 & num2
    images_test_new, labels_test_new = [], []        
    #take threshold samples for digit num1 and digit num2 each
    threshold = 1000
    no1, no2 = 0, 0        
    for i in range(len(labels_test)):
        if no1 < threshold and labels_test[i][0] == num1:
            images_test_new.append(images_test[i])
    #        print (labels_test[i][0])
            labels_test_new.append([1])
            no1 += 1
        elif no2 < threshold and labels_test[i][0] == num2:
            images_test_new.append(images_test[i])
    #        print (labels_test[i][0])
            labels_test_new.append([-1])
            no2 += 1
            
    print (len(images_train_new))
    print (len(labels_train_new))
    
    #for i in range(len(labels_test_new)):
    #    print (labels_test_new[i][0])
    
    images_train_new = np.asarray(images_train_new)
    images_test_new = np.asarray(images_test_new)
    
    labels_train_new = np.asarray(labels_train_new)
    labels_test_new = np.asarray(labels_test_new)
    
    np.random.seed(1)
    #randomly permute data points
    inds = np.random.permutation(images_train_new.shape[0])
    images_train_new = images_train_new[inds]
    labels_train_new = labels_train_new[inds]
    
    inds = np.random.permutation(images_test_new.shape[0])
    images_test_new = images_test_new[inds]
    labels_test_new = labels_test_new[inds]
    
    print (type(images_train_new))
    print (type(labels_train_new))
    return images_train_new, labels_train_new, images_test_new, labels_test_new

# ***************** dataset pre-processing for question (c) ****************

#filter train dataset to contain images for digit 1 with label 1 and images of other digits with labels -1
images_train_multiclass, labels_train_multiclass = [], [] 
#take threshold samples for digit 1 and other digits respectively
threshold = 1000
no1, no2 = 0, 0
for i in range(len(labels_train)):
    if no1 < threshold and labels_train[i][0] == 1:
        images_train_multiclass.append(images_train[i])
#        print (labels_train[i][0])
        labels_train_multiclass.append([1])
        no1 += 1
    elif no2 < threshold and labels_train[i][0] != 1:
        images_train_multiclass.append(images_train[i])
#        print (labels_train[i][0])
        labels_train_multiclass.append([-1])
        no2 += 1

#filter test dataset to contain images for digit 1 with label 1 and images of other digits with labels -1
images_test_multiclass, labels_test_multiclass = [], []        
#take threshold samples for digit 1 and digit 6 each
threshold = 1000
no1, no2 = 0, 0        
for i in range(len(labels_test)):
    if no1 < threshold and labels_test[i][0] == 1:
        images_test_multiclass.append(images_test[i])
#        print ("digit 1",labels_test[i][0])
        labels_test_multiclass.append([1])
        no1 += 1
    elif no2 < threshold and labels_test[i][0] != 1:
        images_test_multiclass.append(images_test[i])
#        print ("other digits",labels_test[i][0])
        labels_test_multiclass.append([-1])
        no2 += 1
        
print (len(images_train_multiclass))
print (len(labels_train_multiclass))

#for i in range(len(labels_test_multiclass)):
#    print (labels_test_multiclass[i][0])

images_train_multiclass = np.asarray(images_train_multiclass)
images_test_multiclass = np.asarray(images_test_multiclass)

labels_train_multiclass = np.asarray(labels_train_multiclass)
labels_test_multiclass = np.asarray(labels_test_multiclass)

np.random.seed(1)
#randomly permute data points
inds = np.random.permutation(images_train_multiclass.shape[0])
images_train_multiclass = images_train_multiclass[inds]
labels_train_multiclass = labels_train_multiclass[inds]

inds = np.random.permutation(images_test_multiclass.shape[0])
images_test_multiclass = images_test_multiclass[inds]
labels_test_multiclass = labels_test_multiclass[inds]

print (type(images_train_multiclass))
print (type(labels_train_multiclass))

# ************ end of dataset pre-processing ***********************



#we are using hinge loss function for svm training
def svm(images_train,labels_train, epochs, lamda):
    # create a weight vector of size equal to features we have and initialize it to zero
    weight = np.zeros(images_train_new.shape[1])
#    print (weight.shape)
    print ("lamda",lamda)
    for epoch in range(1,epochs):
        for i in range(1,len(images_train)):
            #decide learning rate
            learning_rate = 1 / i
            #so, learning rate is higher initially and reduces eventually so that we reach to the point of global minima
            
            #decide regularization weight
#            lamda = 0.00001
            
            if labels_train[i][0] * np.dot(images_train[i], weight) < 1:
                #update the weight by gradient if prediction is not right
                # w <- w + ɳ yn xn
                #also a regularization term is added to shrink weight vector
                weight += learning_rate * ( (images_train[i] * labels_train[i]) - (2 * lamda * weight))
            else:
                #do not change the weight if prediction is right..just add regularization term to shrink weight
                weight -= learning_rate * (2 * lamda * weight)
    return weight
 
def getAcuuracy(images_test, labels_test, weights):
    correct_pred = 0
    for i in range(len(images_test)):
        pred_label = getPrediction(images_test[i], weights)
        if pred_label == labels_test[i][0]:
            correct_pred += 1
    return ( correct_pred * 100) / len(images_test)

def getPrediction(image_test, weights):
    activation = 0
    for image, weight in zip(image_test, weights):
        activation += image * weight
    if activation > 0:
        return 1
    else:
        return -1
  
def getPredictionStatistics(images_test_n, labels_test_n, weight, num1, num2):
    
    correct_pred_num1, total_num1 = 0, 0
    
    for i in range(len(labels_test_n)):
        #for num1, prediction is 1 and actual labels is also 1
        if labels_test_n[i][0] == 1:
            total_num1 += 1
            if getPrediction(images_test_n[i], weight) == 1:
                correct_pred_num1 += 1
    return correct_pred_num1, total_num1

def getConfusion():
    #Initialize confusionmatrix with all 0s
    confMatrix = [[0 for j in range(10)] for i in range(10)]
#    lamda = 0.00001
    
    for i in range(0,10):
        for j in range(0,10):
            if i != j:
                images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(i, j)
                #train svm for binary classification of digit i and digit j
                optWeight = svm(images_train_new, labels_train_new, 5, 0.00001)
                #To update confusion matrix row i, find out the how many predictiosn from class i are classifies into class j i.e incorrect predictions of class i
                correct_pred_num1, total_num1 = getPredictionStatistics(images_test_new, labels_test_new, optWeight, i, j)
                #update diagonal of confusion matrix                
                confMatrix[i][i] += correct_pred_num1
                #update conf(i,j) by number of images that are from category i and are predict-ed as category j
                confMatrix[i][j] += total_num1 - correct_pred_num1
    #Normalize confusion matrix so that rows add to 1
    for conf in confMatrix:
        row_sum = sum(conf)
        for i in range(len(conf)):
            conf[i] = round(conf[i] / row_sum,3)
    return confMatrix

def getPrecision(confMatrix):
    diagonal,conf = 0,0
    for i in range(len(confMatrix)):
        for j in range(len(confMatrix[i])):
            conf += confMatrix[i][j]
            if i == j:
                diagonal += confMatrix[i][j]
                
    return round(diagonal/conf,5)

def getWorstImage(images_test, labels_test, weight):
    images_incorrect = []
    for i in range(len(images_test)):
        if labels_test[i][0] == 1:
            if getPrediction(images_test[i], weight) == -1:
                #append images whose true and predicted label is different
                images_incorrect.append(images_test[i])
    scores = []
    for image in images_incorrect:
        score = 0
        for i in range(len(image)):
            score += abs(weight[i] - image[i])
        scores.append(score)
    ind = scores.index(max(scores))
    return images_incorrect[ind]

#optWeight = svm(images_train_new, labels_train_new, 5)    

#a = getAcuuracy(images_test_new, labels_test_new, optWeight)    
#print (a)   
            
a = input("Enter one of the following question numbers:\n (a) \n (b) \n (c) \n (d) \n (e) \n")

if(a == 'a'):
    #plot of accuracy vs no of epochs
    no_of_epochs = [1, 3, 5, 10, 20]
    accuracy=[]
#    lamda = 0.00001
    images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1, 6)
    for epoch in no_of_epochs:
        optWeight = svm(images_train_new, labels_train_new, epoch, 0.00001)
        accuracy.append(getAcuuracy(images_test_new, labels_test_new, optWeight))    
        
    print (accuracy)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs no of epochs")
    plt.plot(no_of_epochs, accuracy)

elif(a == 'b'):
    #plot of accuracy vs lamda
    lamdas = [0, 0.00001, 0.0001, 0.001, 0.1, 1]
    accuracy=[]
    images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1, 6)
    for l in lamdas:
#        lamda = l
        optWeight = svm(images_train_new, labels_train_new, 5, l)
        accuracy.append(getAcuuracy(images_test_new, labels_test_new, optWeight))    
    print (accuracy)
    plt.xlabel("hyperparameter lamda")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs lamda")
    plt.plot(lamdas, accuracy)
    
elif(a == 'c'):
    #plot of accuracy vs no of epochs for multiclass classification
    no_of_epochs = [1, 3, 5, 10, 20]
    accuracy=[]
#    lamda = 0.00001
    for epoch in no_of_epochs:
        optWeight = svm(images_train_multiclass, labels_train_multiclass, epoch, 0.00001)
        accuracy.append(getAcuuracy(images_test_multiclass, labels_test_multiclass, optWeight))    
    
    print (accuracy)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs no of epochs")
    plt.plot(no_of_epochs, accuracy)
    
elif(a == 'd'):
    confMatrix = getConfusion()
    for row in confMatrix:
        print (row)
    print("Average Precision is:",getPrecision(confMatrix))
    
elif(a == 'e'):
    confMatrix = getConfusion()
    incorrect = []
    for i in range(0,10):
        # get the row for confMatrix excluding diagonal value
        conf = confMatrix[i][0:i] + confMatrix[i][i+1:]
        max_incorrect = max(conf)
        #append incorrectly classified category for digit i
        incorrect.append(conf.index(max_incorrect))
    print(incorrect)
    for i in range(len(incorrect)):
            #now again train svm model to get worst classifi-ed images for digit i
            images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(i,incorrect[i])
            optWeight = svm(images_train_multiclass, labels_train_multiclass, 5, 0.00001)
            #get incorrect/worst image for digit i
            incorrect_image = getWorstImage(images_test_new,labels_test_new, optWeight)
            img_sorted = []
            for j in range(0,len(incorrect_image),28):
                img_sorted.append(incorrect_image[j:j+28])
            p = plt.subplot(2,5,i+1)
            plt.axis('off')
            p.set_title("y:"+str(i) + " yhat:" + str(incorrect[i]))
            plt.imshow(img_sorted,'gray_r')
            