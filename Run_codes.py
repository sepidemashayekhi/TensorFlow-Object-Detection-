from Functions import Detect
modelURL='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
classifilepath='coco.names'
detector=Detect()
detector.readClasses(classifilepath)
# detector.downloadModel(modelURL)