from math import ceil

import cv2 as cv
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import create_bounding_boxes, merge_bounding_boxes 


st.title('Classification of ukrainian handwriting 📝')
st.markdown('Draw a letter or digit. Click ⤓. Repeat!')

left_col, right_col = st.columns(2)
with left_col:
    canvas_result = st_canvas(
        fill_color="rgb(255, 255, 255)",
        stroke_width=5,
        update_streamlit=False,
        background_color="rgb(255, 255, 255)",
        height=300,
        width=300,
    )

canvas_img = canvas_result.image_data[:, :, :3].copy()
if canvas_img.min() == 255:
    right_col.image(canvas_img, 'Your prediction will be here')
    st.warning('You need to draw something first!', icon='⚠️')
    st.stop()

canvas_img_metadata = canvas_result.json_data['objects'] 
if len(canvas_img_metadata) == 1:
    meta = canvas_img_metadata[0]
    left, top = meta['left'], meta['top']
    width, height = meta['width'], meta['height']
    p1 = (int(left), int(top))
    p2 = (ceil(left + width) + 4, ceil(top + height) + 4)
else:
    boxes = create_bounding_boxes(canvas_img_metadata)    
    p1, p2 = merge_bounding_boxes(boxes)
cv.rectangle(canvas_img, p1, p2, (255, 0, 0), 2)
right_col.image(canvas_img, 'Here is your prediction!')
