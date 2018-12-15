import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import operator

#Loading the data
M = loadmat('MNIST_digit_data.mat')

images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
print (images_train.shape)
print (labels_train.shape)

def sortDataset(num1, num2):
    #filter train dataset to contain only images for digit num1 & num2
    images_train_new, labels_train_new = [], [] 
    #take threshold samples for digit num1 and digit num2 each
    threshold = 500
    no1 = 0
    for i in range(len(labels_train)):
        if no1 < threshold and labels_train[i][0] == num1:
            images_train_new.append(images_train[i])
    #        print (labels_train[i][0])
            labels_train_new.append([1])
            no1 += 1
        elif no1 > threshold and no1 < (2 * threshold) and labels_train[i][0] == num2:
            images_train_new.append(images_train[i])
    #        print (labels_train[i][0])
            labels_train_new.append([-1])
            no1 += 1
    
    #filter test dataset to contain only images for digit num1 & num2
    images_test_new, labels_test_new = [], []        
    #take threshold samples for digit num1 and digit num2 each
    threshold = 500
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

def filterDataset(num1, num2, threshold):
    #filter train dataset to contain only images for digit num1 & num2
    images_train_new, labels_train_new = [], [] 
    #take threshold samples for digit num1 and digit num2 each
#    threshold = 500
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
    threshold = 500
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

def getAcuuracy(images_test, labels_test, weights):
    correct_pred = 0
    for i in range(len(images_test)):
        pred_label = predict(images_test[i], weights)
        if pred_label == labels_test[i][0]:
            correct_pred += 1
    return ( correct_pred * 100) / len(images_test)

#this functions predicts the class of one image
def predict(image, weights):
    activation = weights[0]
    for i in range(len(image)):
        activation += weights[i + 1] * image[i]
    if activation >= 0:
        return 1
    else:
        return -1
    
def perceptron(images_train, labels_train, epochs):
    #initialize weight vector with all zeros
    weight = np.zeros(images_train_new.shape[1]+1)
    print (weight.shape)
    #decide learning rate
    learning_rate = 0.001
    for epoch in range(1,epochs):
        for i in range(0,len(images_train)):
            pred = predict(images_train[i], weight)
            error = labels_train[i][0] - pred
            #update bias
            weight[0] += learning_rate * error
            #update weight
            weight[1:] += learning_rate * error * images_train[i]
    return weight

def plotBestWorstImages(positive, negative):
    #positive and negative test images
    positive_imgs, negative_imgs = [], []
    for i in range(len(labels_test_new)):
        if labels_test_new[i][0] == 1:
            positive_imgs.append(images_test_new[i])
        else:
            negative_imgs.append(images_test_new[i])
            
    #now let's find out best and worst images in positive_img and negative_img by finding out their score
    positive_best, positive_worst, negative_best, negative_worst = [], [], [], []
    #score of image depends on weights learned. 
    positive_imgs_score = []
    for i in range (len(positive_imgs)):
        score = 0
        for j in range (len(positive_imgs[i])):
            score += abs(positive[j] - positive_imgs[i][j]) 
        positive_imgs_score.append(score)
    
    #score of image depends on weights learned. 
    negative_imgs_score = []
    for i in range (len(negative_imgs)):
        score = 0
        for j in range (len(negative_imgs[i])):
            score += abs(negative[j] - negative_imgs[i][j]) 
        negative_imgs_score.append(score)
        
    #now, we have positive images and their scores. we need to find best and worst positive images
    #sort scores and then take images corresponding to sorted scores
    positive_imgs_sorted = list(zip(positive_imgs,positive_imgs_score))
    #now sort depending on scores
    positive_imgs_sorted = sorted(positive_imgs_sorted,key=operator.itemgetter(1),reverse = True)

    positive_imgs_sorted_first20 = positive_imgs_sorted[0:20]
    for i in range(len(positive_imgs_sorted_first20)):
        positive_best.append(positive_imgs_sorted_first20[i][0])
    print ("Positive Best 20")   
    for i in range(len(positive_best)):
        p = np.asarray(positive_best[i])
        im = p.reshape((28,28),order='F')  
        p = plt.subplot(4,5,i+1)
        p.set_title("img %d" % (i))
        plt.axis('off')
        plt.imshow(im, cmap="gray_r")
        
    plt.show()
    positive_imgs_sorted_last20 = positive_imgs_sorted[-20:]
    for i in range(len(positive_imgs_sorted_last20)):
        positive_worst.append(positive_imgs_sorted_last20[i][0])
    print ("Positive Worst 20")    
    for i in range(len(positive_worst)):
        p = np.asarray(positive_worst[i])
        im = p.reshape((28,28),order='F')
        p = plt.subplot(4,5,i+1)
        p.set_title("img %d" % (i))
        plt.axis('off')
        plt.imshow(im, cmap="gray_r")
    plt.show()    
        
        
    #sort scores and then take images corresponding to sorted scores
    negative_imgs_sorted = list(zip(negative_imgs,negative_imgs_score))
    #now sort depending on scores
    negative_imgs_sorted = sorted(negative_imgs_sorted,key=operator.itemgetter(1),reverse = True)

    negative_imgs_sorted_first20 = negative_imgs_sorted[0:20]
    for i in range(len(negative_imgs_sorted_first20)):
        negative_best.append(negative_imgs_sorted_first20[i][0])
        
    print ("Negative Best 20")
    for i in range(len(negative_best)):
        p = np.asarray(negative_best[i])
        im = p.reshape((28,28),order='F')
        p = plt.subplot(4,5,i+1)
        p.set_title("img %d" % (i))
        plt.axis('off')
        plt.imshow(im, cmap="gray_r")

    plt.show()
    negative_imgs_sorted_last20 = negative_imgs_sorted[-20:]
    for i in range(len(negative_imgs_sorted_last20)):
        negative_worst.append(negative_imgs_sorted_last20[i][0])
    print ("Negative Worst 20")   
    for i in range(len(negative_worst)):
        p = np.asarray(negative_worst[i])
        im = p.reshape((28,28),order='F')
        p = plt.subplot(4,5,i+1)
        p.set_title("img %d" % (i))
        plt.axis('off')
        plt.imshow(im, cmap="gray_r")
    plt.show()        
def plotWeight():
    positive, negative = [], []
    for w in optWeight[1:]:
        if w >= 0:
            positive.append(w)
        else:
            positive.append(0)
        if w < 0:
            negative.append(w)
        else:
            negative.append(0)
            
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    
    im = positive.reshape((28,28),order='F')
    plt.imshow(im, cmap="gray_r")
    plt.title("Weight vector image for positive weights corresponding to digit 1")
    plt.show()
    
    im = negative.reshape((28,28),order='F')
    plt.imshow(im, cmap="gray")
    plt.title("Weight vector image for negative weights corresponding to digit 6")
    plt.show()
    
    return positive, negative
    
#images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1,6)
a = input("Enter one of the following question numbers:\n (a) \n (b) \n (c) \n (d) \n (e) \n (f)")

if(a == 'a'):
    images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1,6,500)
    no_of_epochs = [1, 3, 4, 10]
    accuracy=[]
    for epoch in no_of_epochs:
        optWeight = perceptron(images_train_new, labels_train_new, epoch)    
        accuracy.append(getAcuuracy(images_test_new, labels_test_new, optWeight))
    print (accuracy) 
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs no of epochs")
    plt.plot(no_of_epochs, accuracy)

if(a == 'b'):
    images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1,6, 500)
    optWeight = perceptron(images_train_new, labels_train_new, 3) 
    plotWeight()
    
elif(a == 'c'):
    images_train_new, labels_train_new, images_test_new, labels_test_new = filterDataset(1,6, 500)
    optWeight = perceptron(images_train_new, labels_train_new, 3) 
    #positive and negative weights
    positive, negative = plotWeight()
    plotBestWorstImages(positive, negative)
   
    
elif (a == 'd'):
    images_train_n, labels_train_n, images_test_n, labels_test_n = filterDataset(1,6, 500)
    print (labels_train_n.shape)
    #length of 10% of training data 
    l = int(labels_train_n.shape[0] / 10)
    print ("length", l)
    #flip the labels for 10% of training data
    indices = np.random.permutation(l)
    for i in indices:
        if labels_train_n[i][0] == 1:
           labels_train_n[i][0] = -1
        else:
           labels_train_n[i][0] = 1
           
    optWeight = perceptron(images_train_n, labels_train_n, 3) 
    positive, negative = plotWeight()
    plotBestWorstImages(positive, negative)
    
elif (a == 'e'):
    images_train_n1, labels_train_n1, images_test_n1, labels_test_n1 = sortDataset(1,6)
    no_of_epochs = [1, 50, 100]
    accuracy=[]
    for epoch in no_of_epochs:
        optWeight = perceptron(images_train_n1, labels_train_n1, epoch)    
        accuracy.append(getAcuuracy(images_test_n1, labels_test_n1, optWeight))
    print (accuracy) 
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs no of epochs")
    plt.plot(no_of_epochs, accuracy)
    
elif (a == 'f'):
    images_train_n2, labels_train_n2, images_test_n2, labels_test_n2 = filterDataset(1,6, 10)
    no_of_epochs = [1, 3, 4, 10]
    accuracy=[]
    for epoch in no_of_epochs:
        optWeight = perceptron(images_train_n2, labels_train_n2, epoch)    
        accuracy.append(getAcuuracy(images_test_n2, labels_test_n2, optWeight))
    print (accuracy) 
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy in %")
    plt.title("accuracy vs no of epochs")
    plt.plot(no_of_epochs, accuracy)