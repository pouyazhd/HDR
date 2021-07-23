'''
Laus Deo 
Digit recognizer project, Final Project of Machine Learning course by Prof M.Sadeghi
Yazd University, Summer 2020
Author:  P.Zahedi
Dataset: MNIST
Part II: Classification
Methode: KNN, NN, Quadratic
'''
import numpy as np
from scipy import stats
import math

def CalcEucledecianDestination(TestVector,TrainVector):
    destination=0
    for i in range(0,len(TestVector)):
      
        destination+=(TestVector[i]-TrainVector[i])**2
    
    return math.sqrt(destination)

def MeanClasses(NumberOfClasses,TrainVector,TrainLabels):
    Class_Count=[]
    TrainMean=[]        
    for i in range(0,NumberOfClasses):
        Class_Count.append(0)
        TrainMean.append([0]*len(TrainVector[0]))
    
    for i in range(0,len(TrainLabels)):
        k=int(TrainLabels[i])
        TrainMean[k]= np.add(TrainMean[k],TrainVector[i]) 
        if i%1000==0:
            print(i,'image summed and ',i/600,'% progressed')
    print(i,'image summed and ',i/600,'% progressed')
    for i in range(0,len(Class_Count)):
        TrainMean[i]=np.divide(TrainMean[i],Class_Count[i]) 
    
    return TrainMean

def NN_Classification(MeanVector,TestVector):
    Dist=[0.0]*len(MeanVector)
    for i in range(0,10):
        Dist[i]=CalcEucledecianDestination(TestVector,MeanVector[i])
    
    return Dist.index(min(Dist))


def KNN(K,TestVector,TrainVector,TrainLabels):
    KNeighbors=[]
    KNeighborLabels=[]
    for i in range(0,K):
        KNeighbors.append([math.inf,math.inf])
        KNeighborLabels.append(math.inf)
    
    for i in range(0,len(TrainVector)):
        Dist=CalcEucledecianDestination(TestVector,TrainVector[i])
        if Dist<=max(KNeighbors)[0]:
            KNeighbors[KNeighbors.index(max(KNeighbors))]=[Dist,TrainLabels[i]]
            KNeighborLabels[KNeighbors.index(max(KNeighbors))]=TrainLabels[i]
  

    return KNeighbors, max(set(KNeighborLabels), key = KNeighborLabels.count)

def QuadraticClassifireParameters(TrainImages,TrainLabels):
    '''
    Calculate mean and covariance of each class
    '''
    ClassSamples = [[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(TrainImages)):
        Class=int(TrainLabels[i])
        ClassSamples[Class].append(TrainImages[i])
 
    TotalMean=[]
    TotalCov=[]
    for i in range(len(ClassSamples)):
        TotalMean.append(np.mean(ClassSamples[i],axis=0))
        TotalCov.append(np.cov(ClassSamples[i],axis=0))
   
    return TotalCov, TotalMean

# Gaussian Classifire
def GaussianPDF(x,mu,sigma):
    print(np.linalg.det(sigma)) # this is shows Quadratic is not working on binary images 
    gx=np.dot((x-mu).transpose,np.linalg.inv(sigma))
    gx=np.dot(gx,(x-mu))
    prob= stats.multivariate_normal(mu,sigma).pdf(x)
    print(gx)
    return prob
    

def GaussianQuadraticClassifier(mu,sigma,TestVector,WeightMatrix):
    for j in range (0,len(TestVector)):
        Probabilitymatrix=[0]*10
        for i in range(0,10):
            Probabilitymatrix[i]=GaussianPDF(TestVector[j],mu[i],sigma[i])*WeightMatrix[i]
            if max(Probabilitymatrix) < WeightMatrix[10]:   
                return -1
            else:
                return Probabilitymatrix.index(max(Probabilitymatrix))

def KFolder(inputArray,trainArray,testArray,K,fold):
    test_bigin = len(inputArray)/K*(fold-1)
    test_end   =  len(inputArray)/K*fold -1
    for i in range(0, len(inputArray) ):
        if(i>=test_bigin and i<=test_end):
            testArray.append(inputArray[i])
        else: 
            trainArray.append(inputArray[i])
