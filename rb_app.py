import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

##########
##### Set up sidebar.
##########

# Add in location to select image.

st.title("Image Analytics for Retail Intelligence  ")
st.write("SKU Item Detection using Computer Vision : Demo")
file = st.file_uploader("Please upload an image file", type=["jpg","jpeg","png",'tif'])


## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)


##########
##### Set up main app.
##########

if file is None:
    st.text("Please upload an image file")
else:
  image = Image.open(file)
  
  ## Subtitle.
  st.write('### Inferenced Image')

  # Convert to JPEG Buffer.
  buffered = io.BytesIO()
  image.save(buffered, quality=90, format='JPEG')

  # Base 64 encode.
  img_str = base64.b64encode(buffered.getvalue())
  img_str = img_str.decode('ascii')

  ## Construct the URL to retrieve image.
  upload_url = ''.join([
      #'https://detect.roboflow.com/',
      #'?model=retail_dataset&version=1&api_key=lKyG2oSGXzoyvK2Zsjkj',
      'https://infer.roboflow.com/retail_dataset',
      '?access_token=lKyG2oSGXzoyvK2Zsjkj',
      '&format=image',
      f'&overlap={overlap_threshold * 100}',
      f'&confidence={confidence_threshold * 100}',
      '&stroke=3',
      '&labels=True'
  ])

  ## POST to the API.
  r = requests.post(upload_url,
                    data=img_str,
                    headers={
      'Content-Type': 'application/x-www-form-urlencoded'
  })

  image = Image.open(BytesIO(r.content))

  # Convert to JPEG Buffer.
  buffered = io.BytesIO()
  image.save(buffered, quality=90, format='JPEG')

  # Display image.
  st.image(image,
           use_column_width=True)

  ## Construct the URL to retrieve JSON.
  upload_url = ''.join([
      #'https://infer.roboflow.com/rf-bccd-bkpj9--1',
      #'?access_token=vbIBKNgIXqAQ'
      'https://infer.roboflow.com/retail_dataset',
      '?access_token=lKyG2oSGXzoyvK2Zsjkj'
  ])

  ## POST to the API.
  r = requests.post(upload_url,
                    data=img_str,
                    headers={
      'Content-Type': 'application/x-www-form-urlencoded'
  })

  ## Save the JSON.
  output_dict = r.json()

  ## Generate list of confidences.
  confidences = [box['confidence'] for box in output_dict['predictions']]

  ## Summary statistics section in main app.
  st.write('### Summary Statistics')
  st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
  st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

  ## Histogram in main app.
  st.write('### Histogram of Confidence Levels')
  fig, ax = plt.subplots()
  ax.hist(confidences, bins=10, range=(0.0,1.0))
  st.pyplot(fig)

