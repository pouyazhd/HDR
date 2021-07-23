'''
Laus Deo 
Digit recognizer project, Final Project of Machine Learning course by Prof M.Sadeghi
Yazd University, Summer 2020
Author : P.Zahedi
Dataset: MNIST
Part II: Classification
Methode: KNN Classifier 
'''
import numpy as np
import sys
import DRP_Preprocessing as prep
import DRP_Classifires as clsfr

def KNNClassifire(k=10):
        #KNN Classifie
    Error=[0]*5
    for fold in range(1,6):
        print('fold:',fold)
        TrainArrayFolding_Data=[]
        TestArrayFolding_Data=[]
        TrainArrayFolding_Label=[]
        TestArrayFolding_Label=[]
        clsfr.KFolder(TrainImages,TrainArrayFolding_Data,TestArrayFolding_Data,5,fold)
        clsfr.KFolder(TrainLabels,TrainArrayFolding_Label,TestArrayFolding_Label,5,fold)
        for i in range(0,len(TestArrayFolding_Label)):
            KNeighbors, ClassifiedAs=clsfr.KNN(k,TestArrayFolding_Data[i],TrainArrayFolding_Data,TrainArrayFolding_Label)
            if i%1000==0:
                print('KNN: ',i*100/len(TestArrayFolding_Data),'% progressed')

        for q in range(0,len(TestArrayFolding_Label)):
            if KNeighbors[q]!=TestArrayFolding_Label[q]:
                Error[fold-1]+=1
        Error[fold-1]/=len(TestArrayFolding_Label)

    print('5 fold cross validataion for 10-NN mean: ',np.mean(Error))
    print('5 fold cross validataion for 10-NN cov: ',np.cov(Error))

def NNClassifire():
        # NN Classification
    # find mean Vector
    Error=[0]*5
    for fold in range(1,6):
        TrainArrayFolding_Data=[]
        TestArrayFolding_Data=[]
        TrainArrayFolding_Label=[]
        TestArrayFolding_Label=[]
        clsfr.KFolder(TrainImages,TrainArrayFolding_Data,TestArrayFolding_Data,5,fold)
        clsfr.KFolder(TrainLabels,TrainArrayFolding_Label,TestArrayFolding_Label,5,fold)
        TrainMean=clsfr.MeanClasses(10,TrainImages,TrainLabels)
        NN_result=[None]*len(TestArrayFolding_Data)
        for i in range(0,len(TestArrayFolding_Data)):
            NN_result[i]=clsfr.NN_Classification(TrainMean,TestArrayFolding_Data[i])
        
        for q in range(0,len(TestArrayFolding_Label)):
            if NN_result[q]!=TestArrayFolding_Label[q]:
                Error[fold-1]+=1
        Error[fold-1]/=len(TestArrayFolding_Label)
    
    print('5 fold cross validataion for NN mean: ',np.mean(Error))
    print('5 fold cross validataion for NN mean:',np.cov(Error))

def QuadraticClassification(NumberOfTests=500,TrainImages=None,TrainLabels=None,TestImages=None,TestLabels=None):
        # Quadratic Classification
    
    # Train
    CovMatrix,MeanMatrix=clsfr.QuadraticClassifireParameters(TrainImages,TrainLabels)
    WeightMatrix=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.99,0.01] # each class has wight 0.1 and non classified is 0.01. total weights are 1.0 
    Probabilitymatrix=[0]*10
    TestClassifie=[-1]*NumberOfTests
    # Test
    for j in range(0,NumberOfTests):
        TestClassifie[j]=clsfr.GaussianQuadraticClassifier(MeanMatrix,CovMatrix,TestImages[j],WeightMatrix)

    Error=len(list(set(TestClassifie) - set(TestLabels[0:NumberOfTests])))/NumberOfTests
    print(Error)

def help():
    help='''
    Digit Recognizer project
    to use this classifire, two argument is needed, dataset address and classification method
    all MNIST datasets should be in a directory which user will address it as first argument and three classification methods are:
     "1" refer to KNNClassifire
     "2" refer to NNClassifire
     "3" which run QuadraticClassification
    '''


if __name__ == "__main__":

    TrainImages=[]
    TrainLabels=[]
    SensitivityLevel=110
    Dataset=sys.argv[1]
    ClassificationMethod=sys.argv[2]

    #open Train database
    TrainDataFileAddress= Dataset+'/train-images-idx3-ubyte.gz'
    TrainImageLabelAddress= Dataset+'train-labels-idx1-ubyte.gz'
    TrainImages, TrainLabels =prep.OpenDataset(TrainDataFileAddress,TrainImageLabelAddress)
    
    #open Test database
    TestDataFileAddress = Dataset+'/t10k-images-idx3-ubyte.gz'
    TestImageLabelAddress = Dataset+'/train-labels-idx1-ubyte.gz'
    TestImages, TestLabels =prep.OpenDataset(TestDataFileAddress,TestImageLabelAddress)

    #binerize image
    TrainImages=prep.Binerizer(TrainImages,SensitivityLevel)
    TestImages=prep.Binerizer(TestImages,SensitivityLevel)
   
    
    if ClassificationMethod=='1':
        KNNClassifire(k=10)
    elif ClassificationMethod=='2':
         NNClassifire()
    elif ClassificationMethod=='3':
        QuadraticClassification(TrainImages=TrainImages, TrainLabels=TrainLabels, TestImages=TestImages, TestLabels=TestLabels)
    else:
        help()

    



