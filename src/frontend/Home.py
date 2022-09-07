import os

import requests
import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from dotenv import load_dotenv

from utils import create_bounding_boxes, merge_bounding_boxes, encode_image

load_dotenv()  # load environmental variables
API_ENDPOINT = os.environ['API_ENDPOINT']
HEIGHT = 170


def add_border(img: np.ndarray) -> np.ndarray:
   return np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=215) 


st.title('Classification of ukrainian handwriting 📝')
st.markdown('Draw a letter or digit. Click ⤓. Repeat!')

left_col, right_col = st.columns(2)
with left_col:
    canvas_result = st_canvas(
        fill_color="rgb(255, 255, 255)",
        stroke_width=9,
        update_streamlit=False,
        background_color="rgb(255, 255, 255)",
        height=HEIGHT,
        width=HEIGHT,
    )

try:
    canvas_img_metadata = canvas_result.json_data['objects'] 
except TypeError:
    st.stop()

if len(canvas_img_metadata) == 0:
    white_img = np.zeros((HEIGHT - 2, HEIGHT - 2, 3), dtype=np.uint8) + 255
    right_col.image(
        add_border(white_img), 'Your prediction will be here', HEIGHT
    )
    st.warning('You need to draw something first!', icon='⚠️')
    st.stop()

canvas_img = canvas_result.image_data[:, :, :3].copy()
boxes = create_bounding_boxes(canvas_img_metadata, 9)    
p1, p2 = merge_bounding_boxes(boxes)

img_with_bb = cv.rectangle(canvas_img.copy(), p1, p2, (255, 0, 0), 2)
right_col.image(
    add_border(img_with_bb[1:-1, 1:-1]), 'Here is your prediction!', HEIGHT
)

cropped_img = canvas_img[p1.y:p2.y, p1.x:p2.x]
padded_img = cv.copyMakeBorder(
    cropped_img, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=[255, 255, 255]
)
encoded_img = encode_image(padded_img)
file = {'img_file': ('test_img.png', encoded_img)}

labels = '0123456789абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
with st.spinner('Processing your image...'):
    response = requests.post(API_ENDPOINT, files=file)
    label_logits, is_upp_logits = response.json().values()
    best, *top2 = np.argsort(-np.array(label_logits))[:3]
    kind = 'a lowercase' if is_upp_logits <= 0.5 else 'an uppercase'
    prediction_msg = (
        f'The model thinks that you drew {kind} {labels[best]}.'
        f'  \nIt might also be: {" ".join(labels[i] for i in top2)}'
    )
st.info(prediction_msg, icon="🤖")
