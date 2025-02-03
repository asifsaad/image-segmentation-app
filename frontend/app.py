# frontend/app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2  # Only if needed for any additional processing

# Import the segmentation function from your backend
import sys
import os

# Adjust the path so that Python can find the backend module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend import segmentation

def main():
    st.title("Image Segmentation App")
    st.write("Upload an image to see its segmentation result.")
    
    # Image uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert the image to a NumPy array for processing
        image_array = np.array(image.convert("RGB"))
        
        if st.button("Segment Image"):
            with st.spinner("Segmenting..."):
                segmented = segmentation.segment(image_array)
                
                # If the segmentation output is a NumPy array, convert it to a PIL Image for display
                segmented_image = Image.fromarray(segmented.astype('uint8'), 'RGB')
                st.image(segmented_image, caption="Segmented Image", use_column_width=True)
                st.success("Segmentation complete!")

if __name__ == "__main__":
    main()
