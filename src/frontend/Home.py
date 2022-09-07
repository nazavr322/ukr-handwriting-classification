from math import ceil

import requests
import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import create_bounding_boxes, merge_bounding_boxes, encode_image


HEIGHT = 170

@st.cache
def create_white_img():
    white_img = np.zeros((HEIGHT, HEIGHT), np.uint8) + 255
    return np.pad(white_img, 1, constant_values=210)


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
    white_img = create_white_img()
    right_col.image(white_img, 'Your prediction will be here', HEIGHT)
    st.warning('You need to draw something first!', icon='⚠️')
    st.stop()

canvas_img = canvas_result.image_data[:, :, :3].copy()
if len(canvas_img_metadata) == 1:
    meta = canvas_img_metadata[0]
    left, top = meta['left'], meta['top']
    width, height = meta['width'], meta['height']
    p1 = (int(left), int(top))
    p2 = (ceil(left + width) + 9, ceil(top + height) + 9)
else:
    boxes = create_bounding_boxes(canvas_img_metadata, 9)    
    p1, p2 = merge_bounding_boxes(boxes)
img_with_bb = cv.rectangle(canvas_img.copy(), p1, p2, (255, 0, 0), 2)
right_col.image(img_with_bb, 'Here is your prediction!', HEIGHT)

# cropped_img = canvas_img[p1[1]:p2[1], p1[0]:p2[0]]
# padded_img = cv.copyMakeBorder(
#     cropped_img, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=[255, 255, 255]
# )
# encoded_img = encode_image(padded_img)
# file = {'img_file': ('test_img.png', encoded_img)}
# response = requests.post('http://127.0.0.1:8000/file', files=file)
# st.write(response.json())
