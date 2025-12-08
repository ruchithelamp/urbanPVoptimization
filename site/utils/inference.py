# utils/inference.py

import numpy as np
import cv2
from ultralytics import YOLO
from utils.conversions import get_city_conversion
import streamlit as st

# MAKE SURE model file is in root of site folder PLEASE
#  model path will be called from site.py; path should be from site.py POV
# we are getting it from SUPABASE and saving to /tmp/

MODEL_PATH = "/tmp/best.pt"

@st.cache_resource
def load_model(_sup):
    # download pt from supabase
    path = download_model_from_supabase(_sup)
    # load YOLO model from that path
    return YOLO(path)




# ---------------------- Inferencing Functions ------------------------
def download_model_from_supabase(supabase, bucket="models", filename="best.pt"):
    """
    get YOLO model from supabase and saves to env
    Returns the local path.
    """
    data = supabase.storage.from_(bucket).download(filename)

    local_path = "/tmp/best.pt"
    with open(local_path, "wb") as f:
        f.write(data)

    return local_path

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
def roofarea(img_path, city, supa): 
  '''
  Docstring for roofarea
  
  :param img_path: image file to inference
  :param city: city code for conversion rules
  out: float area in square feet
  '''
  # Get city conversion (float)
  conversion = get_city_conversion(city)

  # streamlit is messing with image type by the time it gets to the model 
  import tempfile
  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(img_path.read())
    tmp.flush()
    img_path = tmp.name

  # YOLO predict (roof shapes), get Results object
#   from app import supabase 
  model = load_model(_sup=supa)
  predicted = model.predict(img_path, save=False)[0]

  # Inference: Get all the roof pixels
  px = find_roof_shapes(predicted)

  # Find real life Area of all detected roof shapes
  # conversion ft squared times all pixels
  a_sqft = px * (conversion ** 2)

  return a_sqft
