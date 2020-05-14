import numpy as np
import pylab as pl
import math
from sklearn.metrics import confusion_matrix
# Gaussian Naive Baye's Binary Classification for spam data
# Spencer Duncan

#read data and return a np array
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

def seperateData(data, percent):
    entries = data.shape[0]
    percent = percent/100
    index = math.floor(entries*percent)
    train = data[0:index,:]
    test = data[index+1:entries, :]
    return train, test

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
    print(items)
    spam = np.sum(data[:,features-1]) / items * 100
    real = 100-spam
    print(spam)
    print(real)
    return spam, real

def calcstd(spam, real):
    size, features = np.shape(spam)
    std = np.zeros((features,2))
    std[:,0] = np.std(spam,0, float)
    std[:,1] = np.std(real,0, float)
    zeroCond = std == 0
    std[zeroCond] = .0001
    return std

def calcmean(spam, real):
    size, features = np.shape(spam)
    std = np.zeros((features,2))
    std[:,0] = np.mean(spam,0, float)
    std[:,1] = np.mean(real,0, float)
    return std

def N(x, std, mean):
    exp = -(math.pow((x-mean), 2) / (2*math.pow(std,2)))
    val = 1/(math.sqrt(2*math.pi)*std)* math.pow(math.e, exp)
    return val

def 

def classify(data, prior, std, mean):
    size = data.shape[0]
    classProb = 
    for i in range(size):
        labels.append(N)

    return 0


def main():
    filename = "spambase.data"
    data = readdata(filename)
    np.random.shuffle(data)
    
    train, test = seperateData(data, 70)
    trainSpam, trainReal = seperateClass(train)
    
    std = calcstd(trainSpam, trainReal)
    mean = calcmean(trainSpam, trainReal)

    prior = calcprior(data)
    print(prior)

    # now run a test
    test, targets = test[:,:57], test[:,57:58]
    labels = classify(test, prior, std, mean)

main()