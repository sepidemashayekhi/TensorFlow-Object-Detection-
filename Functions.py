import cv2,os,time
import  tensorflow as tf
from tensorflow.python.keras.utils.data_utils import  get_file
import  numpy as np
# print ("finish load frameworks ........")

class Detect:
    def __init__(self):
        pass
    def readClasses(self,Classifilepath):
        with open(Classifilepath, 'r') as  f:
            self.classList = f.read().splitlines()
        self.ColorList=np.random.uniform(low=0,high=255,size=(len(self.classList),3))
        print(' len classlist',len(self.classList),"len color list",len(self.ColorList))

    def downloadModel(self,modelURL):
        filename=os.path.basename(modelURL)
        self.modelName=filename[:filename.index('.')]
        self.cachDir="./pretrained_models"
        os.makedirs(self.cachDir,exist_ok=True)
        get_file(fname=filename,origin=modelURL,
                 cache_dir=self.cachDir,cache_subdir='checkpoints',extract=True)





