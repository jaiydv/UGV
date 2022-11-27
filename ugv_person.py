from turtle import distance
from nbformat import write
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from playsound import playsound
import socket
import time


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  output_dict = model(input_tensor)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  if 'detection_masks' in output_dict:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def alert():
  playsound(r'D:\\REAL_OBJECT\Danger Alarm Sound Effect.mp3')
  print('playing sound using  playsound')
  time.sleep(2)

def show_inference(model, image_path,class_id):
  image_np = image_path
  output_dict = run_inference_for_single_image(model, image_np)
  boxes = []
  classes = []
  scores = []

  for i,x in enumerate(output_dict['detection_classes']):
    if x in class_id and output_dict['detection_scores'][i] > 0.58:
      classes.append(x)
      boxes.append(output_dict['detection_boxes'][i])
      scores.append(output_dict['detection_scores'][i])
  boxes = np.array(boxes)
  classes = np.array(classes)
  scores = np.array(scores)
  print(boxes," ",scores)
  sound=0
  if np.any(scores>0.58):
    x1, x2, y1, y2 = boxes[0][0] *224, boxes[0][1] *224,boxes[0][2]*224, boxes[0][3]*224
    sound=1
    # alert()

  vis_util.visualize_boxes_and_labels_on_image_array(image_np,boxes,classes,scores,category_index,instance_masks=output_dict.get('detection_masks_reframed', None),use_normalized_coordinates=True,line_thickness=2)
  
  return [image_np,sound]


def load_model(model_name):
  

  model_dir = r"ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model"
  
  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

categories = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(categories, use_display_name=True)

model_name='ssd_mobilenet_v2_320x320_coco17_tpu-8'
model=load_model(model_name)

class_id=[]
for i in range(1,91):
  class_id.append(i)
  

  
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

s.bind(("172.16.2.157",9999))

s.listen(3)
print("waiting for connections")

vid_out=r"ugv_result12.mp4"
f=0
while True:
  c,addr=s.accept()

  if c:
    video=cv.VideoCapture(0)
      
    width = int(video.get(3))
    height = int(video.get(4))
    fps = int(video.get(cv.CAP_PROP_FPS))
    codec = cv.VideoWriter_fourcc(*'MJPG') ##(*'XVID')
    out = cv.VideoWriter(vid_out, codec, fps, (width, height))
    while True :
      rec,frame=video.read()

      
      print("waiting for connections")
      detect=show_inference(model,frame,class_id)
      txt=str(detect[1])
      detect=detect[0]

      print("Connnected with ",addr)
      

      c.sendall(bytes(str(txt),"utf-8"))

      out.write(detect)
      detect = cv.putText(detect,str(txt),(100,100),cv.FONT_HERSHEY_SIMPLEX, 1,(225,0,225), 2, cv.LINE_AA)

      cv.imshow("detection",detect)
      
      
      if cv.waitKey(20) & 0xFF==ord('q'):
      
        break

  else:
    break


  video.release()

