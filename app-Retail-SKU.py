import torch
import cv2,os
from PIL import Image,ImageDraw
import streamlit as st
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@st.cache
def load_model():
    model=  torch.hub.load('ultralytics/yolov5', 'custom',
                           path='Retail_100_best.pt'
                           #,force_reload=True
                          )

    return model
model = load_model()
model.conf = 0.40



st.title("Image Analytics for Retail Intelligence  ")
st.write("SKU Item Detection using Computer Vision : Demo")
file = st.file_uploader("Please upload an image file", type=["jpg","jpeg","png",'tif'])

if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file).convert("RGB")
    output=model(image,416)
    df = pd.DataFrame(output.pandas().xyxy[0])
    #st.write(df)
    for i in range(len(df)):
        x,y,w,h = df['xmin'][i],df['ymin'][i],df['xmax'][i]-df['xmin'][i],df['ymax'][i]-df['ymin'][i]
        bboxShape=[(x, y), (x+w, y+h)]
        draw = ImageDraw.Draw(image)
        draw.rectangle(bboxShape, outline='#00FFFF',width=5)
    st.image(image, use_column_width=True)
        
