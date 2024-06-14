# import tkinter as tk
# from tkinter import *
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# import pyttsx3
# import time
# from scipy.stats import mode


# WORKSPACE_PATH = 'RealTimeObjectDetection/Tensorflow/workspace'
# SCRIPTS_PATH = 'RealTimeObjectDetection/Tensorflow/scripts'
# APIMODEL_PATH = 'RealTimeObjectDetection/Tensorflow/models'
# ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
# IMAGE_PATH = WORKSPACE_PATH+'/images'
# MODEL_PATH = WORKSPACE_PATH+'/models'
# PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
# CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
# CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
# CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 



# import tensorflow as tf
# from object_detection.utils import config_util
# # from object_detection.protos import pipeline_pb2
# # from google.protobuf import text_format
# labels = [{'name':'A','id':1},{'name':'B','id':2},{'name':'C','id':3},{'name':'D','id':4},{'name':'E','id':5},{'name':'G','id':6},{'name':'T','id':7}]

# # labels = [{'name':'A','id':1},{'name':'B','id':2}]
# print(labels)
# with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')
# CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
# # print(CONFIG_PATH,"\nTensorflow/workspace/models/my_ssd_mobnet/pipeline.config")
# # config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


# import os
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder

# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

# @tf.function
# def detect_fn(image):
#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     # print(prediction_dict)
#     detections = detection_model.postprocess(prediction_dict, shapes)
#     return detections

# import time
# from scipy.stats import mode 
# last_time_spoken = time.time()

# category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
# data =[]
# characters = []
# word = ""

# last_time_spoken = time.time()
# engine = pyttsx3.init()
# engine.setProperty('rate', 110) 



# # Function to update the label text with the detection result
# def update_label_text(text):
#     label.config(text=text)

# # Function to update the camera feed
# # Function to update the camera feed
# def update_camera_feed():
#     global cap,data,last_time_spoken,characters
#     ret, frame = cap.read()
#     if ret:
#         image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#         detections = detect_fn(input_tensor)
#         num_detections = int(detections.pop('num_detections'))
#         detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
#         # print(num_detections)
#         detections['num_detections'] = num_detections
#         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#         top_detection_class = detections['detection_classes'][0] + 1
#         top_detection_label = category_index[top_detection_class]['name']
#         top_detection_score = detections['detection_scores'][0]
#         # update_label_text(f"{top_detection_label} with confidence {top_detection_score}")
#         update_label_text("-".join(characters))
#         image_np_with_detections = image_np.copy()
#         viz_utils.visualize_boxes_and_labels_on_image_array(
#             image_np_with_detections,
#             detections['detection_boxes'],
#             detections['detection_classes']+1,
#             detections['detection_scores'],
#             category_index,
#             use_normalized_coordinates=True,
#             max_boxes_to_draw=1,
#             min_score_thresh=.5,
#             agnostic_mode=False)
#         data.append(detections['detection_classes'][0] +1)
#         image = cv2.resize(image_np_with_detections, (800, 600))
#         img = Image.fromarray(image)
#         imgtk = ImageTk.PhotoImage(image=img)
#         label_img.imgtk = imgtk
#         label_img.config(image=imgtk)
#         if time.time() - last_time_spoken > 4:
#             # Get the top detected class and its score
#             top_detection_label = category_index[mode(data)[0][0]]['name']
#             print(data,mode(data)[0])
#             data=[]

#             # Convert label and score to text
#             prediction_text = f"{top_detection_label} "
#             # Speak the prediction
#             # engine.say(prediction_text)
#             # engine.runAndWait()

#             last_time_spoken = time.time()
#             characters.append(top_detection_label)
#         # if len(characters) == 3:
#         #     word = "".join(characters)
#         #     characters = []
#         #     engine.say(word)
#         #     engine.runAndWait()
#         #     # engine.startLoop(False)
#         #     update_label_text(word)
#     root.after(33, update_camera_feed) # approx 30 fps

# # Function to start the camera feed
# def start_camera_feed():
#     global cap
#     cap = cv2.VideoCapture(cv2.CAP_DSHOW)
#     cap.set(3, 800)
#     cap.set(4, 600)
#     update_camera_feed()

# # Initialize the Tkinter window
# root = tk.Tk()
# root.title("Real-Time Object Detection")
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# # Set the window size to fullscreen
# root.geometry(f"{screen_width}x{screen_height}+0+0")

# # Initialize the label for displaying detection results
# label = Label(root, text="", font=("Arial", 26))
# label.place(x=screen_width//2, y=650, anchor=tk.CENTER)

# # Initialize the label for displaying the camera feed
# label_img = Label(root)
# label_img.pack()
# start_camera_feed()
# # Add a button to start the camera feed
# # start_button = Button(root, text="Start Camera", command=start_camera_feed)
# # start_button.pack()

# # Start the Tkinter event loop
# root.mainloop()
import tkinter as tk
import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import pyttsx3
import time
from scipy.stats import mode

# Importing TensorFlow and object detection utilities
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import os
# Paths
WORKSPACE_PATH = 'RealTimeObjectDetection/Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
MODEL_PATH = WORKSPACE_PATH + '/models'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'

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
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Label map
labels = [{'name':'A','id':1},{'name':'B','id':2},{'name':'C','id':3},{'name':'D','id':4},{'name':'E','id':5},{'name':'G','id':6},{'name':'T','id':7}]
with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 110)

data = []
characters = []
last_time_spoken = time.time()

# Function to update the label text with the detection result
def update_label_text(text):
    label.configure(text=text)


# Convert PIL Image to CTkImage
def pil_to_ctk_image(pil_image):
    return ctk.CTkImage(light_image=pil_image, size=(800, 600))

# Function to update the camera feed
def update_camera_feed():
    global cap, data, last_time_spoken, characters
    ret, frame = cap.read()
    if ret:
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        top_detection_class = detections['detection_classes'][0] + 1
        top_detection_label = category_index[top_detection_class]['name']
        top_detection_score = detections['detection_scores'][0]
        if top_detection_score>0.6:
            update_label_text("-".join(characters))
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + 1,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.5,
                agnostic_mode=False)
            data.append(detections['detection_classes'][0] + 1)
            image = cv2.resize(image_np_with_detections, (800, 600))
            pil_image = Image.fromarray(image)
            ctk_image = pil_to_ctk_image(pil_image)
            label_img.configure(image=ctk_image)
            # label_img.configure(image=imgtk)
            if len(data)>11:
                # Get the top detected class and its score
                top_detection_label = category_index[mode(data)[0][0]]['name']
                print(data,mode(data)[0])
                data=[]

                # Convert label and score to text
                prediction_text = f"{top_detection_label} "
                print(prediction_text)
                # Speak the prediction
                # engine.say(prediction_text)
                # engine.runAndWait()

                # last_time_spoken = time.time()
                characters.append(top_detection_label)
            if len(characters) == 3:
                word = "".join(characters)
                characters = []
                engine.say(word)
                engine.runAndWait()
                # engine.startLoop(False)
                update_label_text(word)
        else:
            label_img.configure(image= pil_to_ctk_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    root.after(33, update_camera_feed)  # approx 30 fps

# Function to start the camera feed
def start_camera_feed():
    global cap
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cap.set(3, 800)
    cap.set(4, 600)
    start_button.place_forget()
    update_camera_feed()

# Initialize the customtkinter window
ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green
def bg_resizer(e):
    if e.widget is root:
        i = ctk.CTkImage(image, size=(e.width, e.height))
        bg_lbl.configure(text="", image=i)
root = ctk.CTk()
root.title("Real-Time Sign Language Detection")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.wm_attributes('-transparentcolor','grey')
# Set the window size to fullscreen
root.geometry(f"{screen_width}x{screen_height}+0+0")
image = Image.open("RealTimeObjectDetection//R (1).jpeg")
background_image = ctk.CTkImage(image, size=(500, 500))
bg_lbl = ctk.CTkLabel(root, text="", image=background_image)
bg_lbl.place(x=0, y=0)
label_title =ctk.CTkLabel(root,text="Sign Language Detection",font=("Arial",26))
label_title.place(x=screen_width // 2, y=50, anchor=tk.CENTER)

# Initialize the label for displaying detection results
label = ctk.CTkLabel(root, text="", font=("Arial", 26))
label.place(x=screen_width // 2, y=750, anchor=tk.CENTER)

# Initialize the label for displaying the camera feed
label_img = ctk.CTkLabel(root,text="")
label_img.place(x=screen_width//4,y=100)

# Add a button to start the camera feed
start_button = ctk.CTkButton(root, text="Start Camera", command=start_camera_feed,font=("Arial",30))
start_button.place(x=screen_width//2,y=450, anchor=tk.CENTER)
root.bind("<Configure>", bg_resizer)
# Start the customtkinter event loop
root.mainloop()