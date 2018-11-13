import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from collections import defaultdict

#Loading the data
M = loadmat('MNIST_digit_data.mat')

images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
print ('Information about images train dataset')
print (images_train.shape)
print (images_train.dtype.name)
print (type(images_train))
print ('Information about labels train dataset')
print (labels_train.shape)
print (labels_train.dtype.name)
print (type(labels_train))

#just to make all random sequences on all computers the same.
np.random.seed(1)

#randomly permute data points
inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]

#if you want to use only the first 1000 data points.
images_train = images_train[0:10000,:]
labels_train = labels_train[0:10000,:]
#print ('labels_tain',labels_train)
images_test = images_test[0:1000,:]
labels_test = labels_test[0:1000,:]
'''
#show the 10'th train image
i=1
im = images_test[i,:].reshape((28,28),order='F')
plt.imshow(im)
plt.title('Class Label:'+str(labels_test[i][0]))
plt.show()
'''
    
start = time.time()

def kNN(images_train, labels_train, images_test, labels_test, k):
    print ('KNN start',k)
    classDict={}
    l=0      
    total_correct = 0
    avg_acc = 0
    
    for image_test in images_test:       
       arr = labels_test[l]
       #print ('class',arr[0])
       #calculate euclidean distnace between test image and all train images
       classSamples = classDict.setdefault(arr[0],[0,0])
       distances = (images_train - image_test) ** 2
       distances = np.sum(distances,axis=1)
       distances = np.sqrt(distances)      
#       print (distances)
       #sort the euclidean distances and get the indexes for the sorted list
       sorted_distances = distances.argsort()
#       print ('sorted',sorted_distances)   
       #pick first k labels from the sorted list
       k_labels=[]
       i=0
       count=0
       for i in sorted_distances:
           #print ('i',i)
           if(count>=k):
               break
           k_labels.append(labels_train[i,0])
           #print (k_labels[count])
           count += 1   
#       print ('k labels',k_labels)
       # find maximum occuring label as predicted label
       d = defaultdict(int)
       for i in k_labels:
           d[i] += 1
       result = max(d.items(), key=lambda x: x[1])
#       print (result)
#       ind = np.argmax(k_labels)  
#       print (ind)
#       print ('predicted label', result[0])
#       print ('original label',labels_test[l][0]) 
       #compare predicted label with actual test label 
       if result[0] == labels_test[l][0]:
           classDict[arr[0]] = [classSamples[0]+1,classSamples[1]+1]
#           print ('total correact',total_correct)
           total_correct += 1
       else:
           classDict[arr[0]] = [classSamples[0],classSamples[1]+1] 
       avg_acc = (total_correct/(l+1))*100   

       l += 1
#    print (l)
    avg=0.0
    #calculate classwise accuracy
    accuracy = []
    for key in sorted(classDict.keys()):
       arr1 = classDict[key]
       avg += (arr1[0]/arr1[1])*100
       accuracy.append((arr1[0]/arr1[1])*100)              
    
    #return classwise and average accuracy
    return accuracy,avg_acc
       
    
datasamples = np.logspace(np.log10(31.0),np.log10(10000.0),num = 10,base=10.0,dtype='int')


print ('1. Graph of accuracy for different train sample sizes for k = 1\n2. Graph of accuracy for different train sample sizes for varying k\n3. Graph of accuracy for fixed train sample size for varying k')
inp = int(input('Select one of the above: '))

if inp == 1:
    print ('Plot 1')
    #plot 1
    acc = []
    for i in range(10):
        arr,arr1 = kNN(images_train[0:datasamples[i]], labels_train[0:datasamples[i]], images_test, labels_test, 1)
#        print (arr1)
        acc.append(arr1)
    plt.xlabel("Data Samples")
    plt.ylabel("Accuracy")
    plt.plot(datasamples,acc,'-o')
    plt.title('Plot of accuracy vs data samples for differnt train sample sizes and k = 1')
    

elif inp == 2:
    print('Plot 2')
    # plot 2
    k_values = [1,2,3,5,10]
    acc = []
    for k in k_values:
            for i in range(0,len(datasamples)):
                arr,arr1 = kNN(images_train[0:datasamples[i]], labels_train[0:datasamples[i]], images_test, labels_test, k)
                acc.append(arr1)
            plt.plot(datasamples,acc,label='k='+str(k))
            acc=[]
    plt.legend(loc = 'best')
    plt.xlabel("Data Samples")
    plt.ylabel("Accuracy")
    plt.title('Plot of accuracy vs data samples for differnt train sample sizes and different k values')

elif inp == 3:
    print ('Plot 3')
    #plot 3
    k_values  = [1,2,3,5,10]
    acc = []
    
    for k in k_values:
#        print (k)
        arr,arr1 = kNN(images_train[0:1000], labels_train[0:1000], images_train[1001:2000], labels_train[1001:2000], k)    
        acc.append(arr1)
    max1 = np.argmax(acc)
    max1 = k_values[max1]
    print ('Accuracy is maximim for k = ',max1)
    plt.plot(k_values,acc,'-o')
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title('Plot of accuracy vs k values for fixed train sample size and different k values')
    
else:
    print ('No option available')

