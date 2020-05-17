import numpy as np
import pylab as pl
import math
from sklearn.metrics import confusion_matrix, accuracy_score
# Gaussian Naive Baye's Binary Classification for spam data
# Spencer Duncan

#read data and return an np array
def readdata(filename):
    temparray = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(',')
        temparray.append(line)
    data = np.asarray(temparray, float)
    return data

#seperate into train and test sets
def seperateData(data, percent):
    entries = data.shape[0]
    percent = percent/100
    index = math.floor(entries*percent)
    train = data[0:index,:]
    test = data[index+1:entries, :]
    print("Train:", train.shape[0], " Test:", test.shape[0])
    return train, test

#split off the spam from the real datasets
def seperateClass(data):
    features = data.shape[1]
    spamCond = data[:,features-1] == 1
    realCond = data[:,features-1] == 0
    spam = data[spamCond]
    real = data[realCond]
    spam = spam[:,:features-1]
    real = real[:,:features-1]
    return spam, real


def calcprior(data):
    items, features = np.shape(data)
    spam = np.sum(data[:,features-1]) / items * 100
    real = 100-spam
    return spam, real

#calculate the std for each feature in spam and real
def calcstd(spam, real):
    size, features = np.shape(spam)
    std = np.zeros((features,2))
    std[:,0] = np.std(spam,0, float)
    std[:,1] = np.std(real,0, float)
    zeroCond = std == 0
    std[zeroCond] = .0001
    return std

#calculate the mean for each feature in spam and real
def calcmean(spam, real):
    size, features = np.shape(spam)
    std = np.zeros((features,2))
    std[:,0] = np.mean(spam,0, float)
    std[:,1] = np.mean(real,0, float)
    return std

#calculate the p(x|c) 
#if the value gets too small set to small number to prevent underflow
def N(x, std, mean):
    exp = -(math.pow((x-mean), 2) / (2*math.pow(std,2)))
    e = math.pow(math.e, exp)
    if(e == 0.0):
        e = 1e-200
    val = 1/(math.sqrt(2*math.pi)*std)* e
    return val

#sum the log of the propabilities
def logPi(data, std, mean, prior):
    size, features = np.shape(data)
    classProb = []
    for email in range(size):
        spamProb = prior[0]
        realProb = prior[1]
        for feature in range(features):
            spamProb += math.log(N(data[email,feature], std[feature,0], mean[feature,0]))
            realProb += math.log(N(data[email,feature], std[feature,1], mean[feature,1]))
        classProb.append([spamProb, realProb])
    return np.asarray(classProb)

def classify(data, prior, std, mean):
    classprob = logPi(data, std, mean, prior)
    #columns denoting class need to be swapped.
    classprob[:,[0,1]] = classprob[:,[1,0]]
    labels = np.argmax(classprob, axis=1)
    return labels


def main():
    filename = "spambase.data"
    data = readdata(filename)
    np.random.shuffle(data)
    
    train, test = seperateData(data, 70)
    trainSpam, trainReal = seperateClass(train)
    
    #calc prob. model
    std = calcstd(trainSpam, trainReal)
    mean = calcmean(trainSpam, trainReal)

    prior = calcprior(data)

    # now run the test
    test, targets = test[:,:57], test[:,57:58]
    labels = classify(test, prior, std, mean)
    confusion_matrix()
    confusion = confusion_matrix(targets, labels)
    print("Confusion matrix:\n", confusion)
    accuracy = accuracy_score(targets, labels)
    print("Accuracy: ", int(accuracy*100), "%")

main()