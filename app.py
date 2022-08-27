# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:56:48 2022

@author: deepak
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model
model = load_model('Signature_Verification_GDPS_SGD_Novel-FINAL-63-0.66.h5')
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Univeristy</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Deepak Moud, PHD Scholar(2018PUSCEPHDE07061)</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.write("""
        Offline Signature Features Identification using deep Convolutional Neural Network
         """
         )
file= st.file_uploader("Please upload signare for verification", type=("jpg", "png"))
import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data, model):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  block4_pool_features = model.predict(img_reshape)
  print(block4_pool_features)
  if(block4_pool_features> 0.5):
    a= 'Genuine'
  elif(block4_pool_features< 0.5):
    a='Forged'
  return a
  model = load_model("/content/drive/My Drive/Colab Notebooks/Signature Verification Inceptionv3 Model/FINAL/model_inception.h5")

if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  #imagefile.save('uploads/' + secure_filename(imagefile.filename))
  # Save the file to ./uploads        
  #file_path = os.path.join('/content','uploads', secure_filename(f.filename))
  st.image(image, use_column_width=True)
  
if st.button("Predict-VGG16"):
  result=import_and_predict(image,model)
  st.success('VGG16 Model has predicted signature is {}'.format(result))

if st.button("About"):
  st.text(" Deepak Moud")
  st.text("Under the Guidance of Dr. Rekha jain")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Objective 1: Evalution of pre trained Model for signature verification</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)