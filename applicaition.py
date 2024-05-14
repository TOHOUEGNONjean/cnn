#coding :utf-8
import streamlit as lt
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
@lt.cache_data()
def load():
    mo = 'meill_model.h5'
    model = load_model(mo)
    return model
model = load()

def prediction(upload):
    img = Image.open(upload)
    img = np.asarray(img)
    img = cv2.resize(img, (224,224))
    final = np.expand_dims(img, axis=0)
    final = model.predict(final)
    return final[0][0]

lt.title("Notre poubelle intélligente")

upload = lt.file_uploader("Charger l'image que vous voulons classée ", type =['jpeg', 'jpg', 'png'])
c1, c2  = lt.columns(2)
if upload:
    c1.write(Image.open(upload))
    pred = prediction(upload) * 100
    if pred > 50:
        c2.write(f"Je suis certain à {pred} % que l'objet est Recyclable")
    else:
        c2.write(f"Je suis certain à {100 - pred} % que l'objet est Organique")
   