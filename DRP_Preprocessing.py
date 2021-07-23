'''
Laus Deo 
Digit recognizer project, Final Project of Machine Learning course by Prof M.Sadeghi
Yazd University, Summer 2020
Author:  P.Zahedi
Dataset: MNIST
Part II: Preprocessing and open files
'''

import numpy as np
import gzip
import matplotlib.pyplot as plt

def DataParameters(fid):
    '''
    Find count, hight and width of images in dataset. 
    '''
    # read headers 
    magicnumber=np.frombuffer(fid.read(4),dtype=np.uint8)
    Image_count=np.frombuffer(fid.read(4),dtype=np.uint8)
    Image_height=np.frombuffer(fid.read(4),dtype=np.uint8)
    Image_width=np.frombuffer(fid.read(4),dtype=np.uint8)

    # find the size of image 
    Image_count=Image_count[3]+256*Image_count[2]+256**2*Image_count[1]+256**3*Image_count[0]
    Image_height=Image_height[3]+256*Image_height[2]+256**2*Image_height[1]+256**3*Image_height[0]
    Image_width=Image_width[3]+265*Image_width[2]+256**2*Image_width[1]+256**3*Image_width[0]

    return Image_count, Image_height, Image_width

def LabelParameters(fid):
    Label_magicnumber=np.frombuffer(fid.read(4),dtype=np.uint8)
    Label_Image_count=np.frombuffer(fid.read(4),dtype=np.uint8)
    Label_Image_count=Label_Image_count[3]+256*Label_Image_count[2]+256**2*Label_Image_count[1]+256**3*Label_Image_count[0]
    return Label_Image_count

def OpenDataset(DataFile,LabelFile):
    # open training dataset and Labels

    # read images
    fid = gzip.open(DataFile,'r')
    Image_count, Image_height, Image_width = DataParameters(fid)
    buff=np.frombuffer(fid.read(Image_count*Image_height*Image_width),dtype=np.uint8)
    buff=buff.reshape(Image_count,Image_width*Image_height)

    # read lables 
    fid=gzip.open(LabelFile,'r')
    Label_Image_count=LabelParameters(fid)
    Label_Image =np.frombuffer(fid.read(Label_Image_count),dtype=np.uint8)
    
    return buff,Label_Image

    # binerize images
def Binerizer(Images, SensitivityLevel):
    return (Images<SensitivityLevel).astype(np.uint8)*255 # binerizing images    


def showImage(Image):
    image=np.asarray(Image).squeeze()
    plt.imshow(image)
    plt.show()


# draw bouning box
def drawboundingbox(Image,border):
    for i in range(border[1],border[0]+1):
        Image[i,border[2]]=127
        Image[i,border[3]]=127

    for j in range(border[3],border[2]):
        Image[border[1],j]=127
        Image[border[0],j]=127    

    return Image

# flood fill algorithm
def floodfill(Image, x, y,wall_color,set_color,target_color,FloodedAddress):
    if Image[x][y] == target_color:  
        Image[x][y] = set_color
        FloodedAddress.append([x,y])

        #check borders and recursively do it
        if x-1>0:
            if Image[x-1][y]==target_color: # N
                floodfill(Image,x-1, y,wall_color,set_color,target_color,FloodedAddress)
        
        if x-1>0 and y+1<len(Image):
            if Image[x-1][y+1]==target_color: # NE
                floodfill(Image,x-1, y+1,wall_color,set_color,target_color,FloodedAddress)
        if y+1<len(Image):
            if Image[x][y+1]==target_color: # E
                floodfill(Image,x, y+1,wall_color,set_color,target_color,FloodedAddress)
        
        if x+1<len(Image) and y+1<len(Image) :
            if Image[x+1][y+1]==target_color: # SE
                floodfill(Image,x+1, y+1,wall_color,set_color,target_color,FloodedAddress)
        
        if x+1<len(Image):
            if Image[x+1][y]==target_color: # S
                floodfill(Image,x+1, y,wall_color,set_color,target_color,FloodedAddress)
        if x+1<len(Image) and y>0:
            if Image[x+1][y-1]==target_color: # SW
                floodfill(Image,x+1, y-1,wall_color,set_color,target_color,FloodedAddress)
        if y>0:
            if Image[x][y-1]==target_color: # W
                floodfill(Image,x, y-1,wall_color,set_color,target_color,FloodedAddress)
        if x>0 and y>0:
            if Image[x-1][y-1]==target_color: # NW
                floodfill(Image,x-1, y-1,wall_color,set_color,target_color,FloodedAddress)

# looking for holes in binary image. 
# this algorithm will loking for other background points in picture
# the if these vectors have no common coordinates then we have holes        
def FindHoles(Image,background,foreground):
    FilledVectors=[]
    for i in range(0,len(Image)):
        for j in range (0,len(Image)):
            
            if(Image[i][j]==background):
                FloodedAddress=[]
                floodfill(Image,i,j,foreground,70,background,FloodedAddress)
                FilledVectors.append(FloodedAddress)
                
    HolesNumber=len(FilledVectors)-1
    return HolesNumber

# find border
def bouningbox(TrainVector):
    BorderAngels=[]
    Image_width=28
    Image_height=28

    for k in range(0,len(TrainVector)):
        Ymax=-1
        Ymin=-1
        Xmax=-1
        Xmin=-1
        
        for i in range (0,Image_width):
            R=0
            L=0
            for j in range (0,Image_height):
                R=R+int(TrainVector[k,i,j])
                L=L+int(TrainVector[k,Image_width-1-i,j])

            if R<Image_width*255 and Ymin==-1:
                Ymin=i-1
                
            if L<Image_width*255 and Ymax==-1:
                Ymax=Image_height-i
                
        for j in range (0,Image_height):
            R=0
            L=0
            for i in range (0,Image_width):
                R=R+int(TrainVector[k,i,j])
                L=L+int(TrainVector[k,i,Image_height-1-j])            
            if R<Image_height*255 and Xmin==-1:
                Xmin=j-1
                
            if L<Image_height*255 and Xmax==-1:
                Xmax=Image_width-j   

        border=[Ymax,Ymin,Xmax,Xmin]
        showImage(drawboundingbox(TrainVector[k],border)) # uncomment this line to show drawed images
        BorderAngels.append(border)
    return BorderAngels

# calculate blackness ratio
def BlacknessRatioCalc(border,DataVector):
    Ymax,Ymin, Xmax, Xmin=border 
    Blacknessratio=[]
    sum_bounded=0
    sum_bounded_max=(Xmax-Xmin+1)*(Ymax-Ymin+1)*255
    for k in range(0,len(DataVector)):
        for i in range (Ymin,Ymax+1):
            for j in range (Xmin,Xmax+1):
                sum_bounded = sum_bounded+DataVector[k,i,j]

        Blacknessratio.append(1-sum_bounded/sum_bounded_max)
    return Blacknessratio

# find weight in each regon
def RegionWeight(Image,border):
    Ymax,Ymin, Xmax, Xmin=border 
    Xm=int((border[2]+border[3])/2)
    Ym=int((border[0]+border[1])/2)
    UpLeft=0
    UpRight=0
    DownLeft=0
    DownRight=0
    
    for i in range(Xmin,Xm):
        for j in range (Ymin,Ym):
            UpLeft=UpLeft+int(Image[i,j])

    for i in range (Xm,Xmax+1):
        for j in range (Ymin,Ym):
            UpRight = UpRight+int(Image[i,j])

    for i in range (Xm,Xmax+1):
        for j in range (Ym,Ymax):
            DownRight = DownRight+int(Image[i,j])
    
    for i in range(Xmin,Xm):
        for j in range (Ym,Ymax):
            DownLeft=DownLeft+int(Image[i,j])

