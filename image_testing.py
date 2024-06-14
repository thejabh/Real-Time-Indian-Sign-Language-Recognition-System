WORKSPACE_PATH = 'RealTimeObjectDetection/Tensorflow/workspace'
SCRIPTS_PATH = 'RealTimeObjectDetection/Tensorflow/scripts'
APIMODEL_PATH = 'RealTimeObjectDetection/Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


labels = [{'name':'A','id':1},{'name':'B','id':2},{'name':'C','id':3},{'name':'D','id':4},{'name':'E','id':5},{'name':'G','id':6},{'name':'T','id':7}]

# labels = [{'name':'A','id':1},{'name':'B','id':2}]
print(labels)
# with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')
# CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
# print(CONFIG_PATH,"\nTensorflow/workspace/models/my_ssd_mobnet/pipeline.config")
# config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    # print(prediction_dict)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 
import numpy as np
# import pyttsx3
# import time
# from scipy.stats import mode 
# last_time_spoken = time.time()
# engine = pyttsx3.init()
# engine.setProperty('rate', 150) 

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
data =[]
dir="D:/College/Final Year Project/Retrying/RealTimeObjectDetection/test_images"
images_paths = os.listdir(dir)
# Setup capture
# IMAGE_PATH = 'RealTimeObjectDetection/test_images/A_cam.jpg'  # Path to your test image
for IMAGE_PATH in images_paths:
    # Read the image
    image_np = cv2.imread(dir+"/"+IMAGE_PATH)

    # Process the image
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    top_detection_class = detections['detection_classes'][0] +1
    top_detection_score = detections['detection_scores'][0]
    print(top_detection_class,"with confidence",top_detection_score)
    # Convert class to label
    top_detection_label = category_index[top_detection_class]['name']
    # print(top_detection_label)
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    # Display the image with detections
    # cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    print(IMAGE_PATH,top_detection_label)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
