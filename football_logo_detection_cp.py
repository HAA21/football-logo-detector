#cloning the master branch of the Tensorflow Models repository
!git clone https://github.com/tensorflow/models.git

# Commented out IPython magic to ensure Python compatibility.
#compiling the protos
# %cd /content/models/research
!protoc object_detection/protos/*.proto --python_out=.

#install the tensorflow API
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .

# Commented out IPython magic to ensure Python compatibility.
#cloning the repository for access to processed data
# %cd /content
!git clone https://github.com/HAA21/football-logo-detector.git

#defining paths to training data, testng data, and label map
train_record_path = '/content/football-logo-detector/train.record'
test_record_path = '/content/football-logo-detector/test.record'
labelmap_path = '/content/football-logo-detector/labelmap.pbtxt'

"""configuration for training"""

#setting hyperparameters
batch_size = 16
num_steps = 1000
num_eval_steps = 100

#downloading EfficientDet model to colab from tensorflow object detection model zoo
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
!tar -xf efficientdet_d0_coco17_tpu-32.tar.gz

#path to the model checkpoint 
fine_tune_checkpoint = 'efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0'

#downloading the base configuration of the model to colab
!wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
base_config_path = 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config'

#changing the base configuration of the model to point to our custom data
import re

with open(base_config_path) as f:
    config = f.read()

with open('model_config.config', 'w') as f:

  #setting new labelmap path
  config = re.sub('label_map_path: ".*?"', 
             'label_map_path: "{}"'.format(labelmap_path), config)
  
  #setting fine_tune_checkpoint path
  config = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
  
  #setting train tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                  'input_path: "{}"'.format(train_record_path), config)
  
  #setting test tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                  'input_path: "{}"'.format(test_record_path), config)
  
  #setting number of classes.
  config = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(4), config)
  
  #setting batch size
  config = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), config)
  
  #setting training steps
  config = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(num_steps), config)
  
  #setting fine-tune checkpoint type to detection
  config = re.sub('fine_tune_checkpoint_type: "classification"', 
             'fine_tune_checkpoint_type: "{}"'.format('detection'), config)
  
  f.write(config)

model_dir = 'training/'                                                         #log directory that your training process will create
pipeline_config_path = 'model_config.config'                                    #path to our pipeline config file

"""training"""

#training the model with all specifications
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}

#exporting the training output into a savedmodel format for inference
output_directory = 'inference_graph'

!python /content/models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir {model_dir} \
    --output_directory {output_directory} \
    --pipeline_config_path {pipeline_config_path}

#downloading the saved model
from google.colab import files
files.download(f'/content/{output_directory}/saved_model/saved_model.pb')

"""testing"""

# Commented out IPython magic to ensure Python compatibility.
#importing dependencies
import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# %matplotlib inline

#puts image into numpy array to feed into tensorflow graph
def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size

  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#to access the label map to map inference or prediction to label
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

#loading the saved model
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'/content/{output_directory}/saved_model')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)                                           #converts image into tensor
  input_tensor = input_tensor[tf.newaxis,...]                                          #converts the image into batch format

  model_fn = model.signatures['serving_default']                                       #running the inference
  output_dict = model_fn(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))                              #accessing the tensor with prediction
  output_dict = {key:value[0, :num_detections].numpy()                                 #converting tensor into numpy array and taking the 0th index
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64) #converting detected classes to int
   
  if 'detection_masks' in output_dict:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(             #reframing the the bbox mask to the image size
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

#running inference on all images in the given path, and displaying bounding boxes
for image_path in glob.glob('/content/*.jpg'):
  image_np = load_image_into_numpy_array(image_path)
  output_dict = run_inference_for_single_image(model, image_np)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  display(Image.fromarray(image_np))

