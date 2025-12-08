# utils/inference.py

import numpy as np
import cv2
from ultralytics import YOLO
from utils.conversions import get_city_conversion
import streamlit as st

# MAKE SURE model file is in root of site folder PLEASE
#  model path will be called from site.py; path should be from site.py POV

MODEL_PATH = "best.pt"

# Initiate YOLO model from best.pt path

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# ---------------------- Inferencing Functions ------------------------

# find_roof_shapes
def find_roof_shapes(predicted, og_size = (256, 256)): 
  # IN: predicted is instance of ultralytics.engine.results.Results
  # OUT: number of roof pixels

  if predicted.masks is None:
    return 0
  
  pixels = 0

  if predicted.masks is None: 
    return 0
  for mask, class_id in zip(predicted.masks.data, predicted.boxes.cls):
    if int(class_id) == 1:
      # get tensor from memory
      roof_px = mask.cpu()
      # ok thats giving us probabilities, so convert it
      roof_px = (roof_px > 0.5).float()

      # YOLO may be upscaling image for better preds -> influx of pixels
      # change image size back to 256 for good measure, using OPENCV resize
      roof_px_resized = cv2.resize(roof_px.numpy(), 
                                   og_size, 
                                   interpolation=cv2.INTER_NEAREST)
      pixels += np.sum(roof_px_resized)

  return pixels

# roofarea
def roofarea(img_path, city): 
  '''
  Docstring for roofarea
  
  :param img_path: image file to inference
  :param city: city code for conversion rules
  out: float area in square feet
  '''
  # Get city conversion (float)
  conversion = get_city_conversion(city)

  # YOLO predict (roof shapes), get Results object
  predicted = model.predict(img_path, save=False)[0]

  # Inference: Get all the roof pixels
  px = find_roof_shapes(predicted)

  # Find real life Area of all detected roof shapes
  # conversion ft squared times all pixels
  a_sqft = px * (conversion ** 2)

  return a_sqft
