'''
Laus Deo 
Digit recognizer project, Final Project of Machine Learning course by Prof M.Sadeghi
Yazd University, Summer 2020
Author : P.Zahedi
Dataset: MNIST
Part I : Feature Extraction
Methode: KNN Classifier 
'''

# libraries
import DRP_Preprocessing as prep
import DRP_Classifires as clsfr
import sys

if __name__ == "__main__":
    
    TrainImages=[]
    TrainLabels=[]
    SensitivityLevel=110
    Dataset=sys.argv[1]

    #open Train database
    TrainDataFileAddress= Dataset+'/train-images-idx3-ubyte.gz'
    TrainImageLabelAddress= Dataset+'train-labels-idx1-ubyte.gz'
    TrainImages, TrainLabels =prep.OpenDataset(TrainDataFileAddress,TrainImageLabelAddress)
   
    #binerize image
    TrainImages=prep.Binerizer(TrainImages,SensitivityLevel)

    # #find holes
    background=255
    foreground=0
    for i in range(0,100):
        print(prep.FindHoles(TrainImages[i],background,foreground))
        prep.showImage(TrainImages[i])

    # # boundingbox
    BorderAngels=prep.bouningbox(TrainImages)

    # calculate mean vector
    TrainImagesMean = TrainImages.reshape(60000,28*28)
    meanMatrix=clsfr.MeanCalculator(TrainImagesMean,TrainLabels)
    for i in range(0,10):
        prep.showImage(meanMatrix[i].reshape(28,28))

    