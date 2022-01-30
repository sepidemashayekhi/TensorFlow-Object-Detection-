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

    def loadModel(self):
        print ('Loadding Model'+self.modelName)
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cachDir,"checkpoints", self.modelName,'saved_model'))
        print("Model"+self.modelName+"Load is finish")

    def creatbondingBox(self,image,threshold=0.5):
        inputTensor=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor)
        inputTensor=inputTensor[tf.newaxis,...]
        detection=self.model(inputTensor)
        bboxs=detection['detection_boxes'][0].numpy()
        classIndexes=detection['detection_classes'][0].numpy().astype(np.int32)
        classeScores=detection['detection_scores'][0].numpy()

        imH, imW, imC=image.shape
        bboxInd=tf.image.non_max_suppression(bboxs,classeScores,max_output_size=50,
                                             iou_threshold=threshold,score_threshold=threshold)

        if len(bboxInd) !=0:
            for i in bboxInd:
                bbox=tuple(bboxs[i].tolist())
                classConfidence=round(100*classeScores[i])
                classIndex=classIndexes[i]
                classLabelText=self.classList[classIndex-1]
                classColor=self.ColorList[classIndex]
                displayText='{} :{} %'.format(classLabelText,classConfidence)
                ymin,xmin,ymax,xmax=bbox
                ymin, xmin, ymax, xmax=(ymin*imH,xmin*imW,ymax*imH,xmax*imW)
                ymin, xmin, ymax, xmax=int(ymin),int(xmin),int(ymax),int(xmax)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classColor,thickness=1)
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,classColor,2)

                lineWidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))

                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),color=classColor,thickness=5)
                cv2.line(image, (xmin, ymin), (xmin , ymin+lineWidth), color=classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), color=classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin+ lineWidth), color=classColor, thickness=5)
                #################################
                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), color=classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax -lineWidth), color=classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), color=classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), color=classColor, thickness=5)


            return  image
    def predict_image(self,imagePath,threshold=0.5):
        image=cv2.imread(imagePath)
        bboxImage=self.creatbondingBox(image,threshold)
        cv2.namedWindow('result',cv2.WINDOW_NORMAL)
        cv2.imshow("result",bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()







