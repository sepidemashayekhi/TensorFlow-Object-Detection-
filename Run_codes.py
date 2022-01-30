import time

from Functions import Detect
import cv2
# modelURL='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
modelURL='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
# modelURL='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
classifilepath='coco.names'
threshold=0.5
detector=Detect()
detector.readClasses(classifilepath)
detector.downloadModel(modelURL)
detector.loadModel()
imagePath='images/test/2.jpg'
detector.predict_image(imagePath,threshold)

# real Time
cam=cv2.VideoCapture(0)
startTime=0
while True:
    nowTime=time.time()
    fsp=1/(nowTime-startTime)
    success,frame=cam.read()
    Imagebbox=detector.creatbondingBox(frame,threshold=0.5)
    cv2.putText(Imagebbox,"FPS"+str(int(fsp)),(20,70),cv2.FONT_HERSHEY_SIMPLEX,
                1,color=(0,0,255),thickness=2)

    cv2.imshow("frame",Imagebbox)
    if cv2.waitKey(1)==27:
        break


