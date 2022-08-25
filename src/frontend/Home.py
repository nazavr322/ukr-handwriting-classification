from math import ceil

import cv2 as cv
import streamlit as st
from streamlit_drawable_canvas import st_canvas


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
        width=300
    )

canvas_img = canvas_result.image_data[:, :, :3].copy()
if canvas_img.min() == 255:
    right_col.image(canvas_img, 'Your prediction will be here')
    st.warning('You need to draw something first!', icon='⚠️')
    st.stop()
for meta in canvas_result.json_data['objects']:
    left, top = meta['left'], meta['top']
    width, height = meta['width'], meta['height']
    xb, yb = ceil(left + width), ceil(top + height)
    test_img = cv.rectangle(
        canvas_img, (int(left), int(top)), (xb + 4, yb + 4), (255, 0, 0), 1
    )
right_col.image(test_img, 'Here is your prediction')
