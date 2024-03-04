import cv2
import streamlit as st

# Read the image
image = cv2.imread('asphalt-crack.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying median blur to remove salt-and-pepper noise
median_blur = cv2.medianBlur(gray, 9)

# Applying Gaussian blur to remove Gaussian noise
gaussian_blur = cv2.GaussianBlur(image, (7, 7), 0)

# Apply bilateral filter to remove Gaussian and preserve edges
bilateral_filtered = cv2.bilateralFilter(gaussian_blur, 5, 6, 6)

# Display the original and denoised images
st.image(image)
st.image(bilateral_filtered)

