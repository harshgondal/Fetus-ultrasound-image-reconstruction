import streamlit as st
import cv2
import os
import numpy as np
from script import adaptive_median_filter, hybrid_filter_p1, load_images, nafsm_filter, resizeImage, gaussian_filter, segment_image, apply_convex_hull, extract_femur_length, fetus_height, srad_filter, wiener_filter,cal_gest

image_dir = './dataset/images'
path = './results'
images = load_images(image_dir)

st.title("Ultrasound Image Processing")

image_selection = st.selectbox("Select an image to process", range(len(images)))
selected_image = images[image_selection]

st.subheader("Original Image")
with st.container():
    col1 = st.columns(1)[0]
    col1.image(selected_image, caption="Original Image", width=512)

resized_image = resizeImage(selected_image)

filtered_image = resized_image.copy()

st.subheader("Select a Noise Reduction Filter")
filter_choice = st.selectbox("Choose a filter", ["Gaussian Filter","NAFSM Filter", "SRAD Filter","Wiener Filter", "Adaptive Median Filter", "Hybrid Filter NAFSM+SRAD"])

# Apply the selected filter
if filter_choice == "NAFSM Filter":
    nafsm_threshold = st.slider("Set threshold for NAFSM Filter", min_value=0, max_value=255, value=25)
    kernel_size = st.slider("Set initial kernel size for NAFSM Filter", min_value=3, max_value=25, value=7, step=2)
    filtered_image = nafsm_filter(resized_image, threshold=nafsm_threshold, window_size=kernel_size)
    st.image(filtered_image, caption="Filtered Image - NAFSM Filter", width=512)

elif filter_choice == "SRAD Filter":
    iterations = st.slider("Set iterations for SRAD Filter", min_value=1, max_value=20, value=10)
    kappa = st.slider("Set kappa for SRAD Filter", min_value=10, max_value=100, value=30)
    gamma = st.slider("Set gamma for SRAD Filter", min_value=0.01, max_value=1.0, value=0.1)
    filtered_image = srad_filter(resized_image, iterations=iterations, kappa=kappa, gamma=gamma)
    st.image(filtered_image, caption="Filtered Image - SRAD Filter", width=512)

elif filter_choice == "Wiener Filter":
    kernel_size = st.slider("Set kernel size for Wiener Filter", min_value=3, max_value=15, value=7)
    filtered_image = wiener_filter(resized_image, kernel_size=kernel_size)
    st.image(filtered_image, caption="Filtered Image - Wiener Filter", width=512)

elif filter_choice == "Adaptive Median Filter":
    kernel_size = st.slider("Set initial kernel size for Adaptive Median Filter", min_value=3, max_value=15, value=3, step=2)
    max_kernel_size = st.slider("Set maximum kernel size for Adaptive Median Filter", min_value=3, max_value=15, value=7, step=2)
    filtered_image = adaptive_median_filter(resized_image, kernel_size=kernel_size, max_kernel_size=max_kernel_size)
    st.image(filtered_image, caption="Filtered Image - Adaptive Median Filter", width=512)

elif filter_choice == "Hybrid Filter NAFSM+SRAD":
    filtered_image = hybrid_filter_p1(resized_image)
    st.image(filtered_image, caption="Filtered Image - Hybrid Filter NAFSM+SRAD", width=512)

elif filter_choice=="Gaussian Filter":
    kernel_size = st.slider("Set initial kernel size for Adaptive Median Filter", min_value=3, max_value=19, value=7, step=2)
    filtered_image=gaussian_filter(resized_image, kernel_size=kernel_size)
    st.image(filtered_image, caption="Filtered Image - Gaussian Filter", width=512)


st.subheader("Threshold Adjustment")

# if st.button("Process with seleted filter"):
with st.container():
    threshold = st.slider("Threshold Value", min_value=0, max_value=255, value=125)
    thresholded_img = segment_image(filtered_image, threshold)
    st.image(thresholded_img, caption=f"Thresholded Image at {threshold}", width=512, )

if st.button("Process with Selected Threshold"):
    # segmented_img = segment_image(thresholded_img)
    
    contours, convex_hull_img = apply_convex_hull(thresholded_img)
    
    femur_length, point1, point2, marked_image = extract_femur_length(convex_hull_img, resized_image)
    marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)

    name = f"processed_image_{image_selection}"
    
    st.subheader("Segmented Image with Convex Hull")

    with st.container():
        st.image(convex_hull_img, caption="Convex Hull", width=512)
        
    st.subheader("Marked Image with Femur Length")

    with st.container():
        st.image(marked_image, caption="Marked Image with Femur Length", channels="RGB", width=512)
        st.write(f"Femur Length: {femur_length:.2f} pixels")
        
    if femur_length:
        
        fetus_height_cm, length_in_cm = fetus_height(femur_length)
        st.header("Results")
        col6, col7 = st.columns(2)
        
        with col6:
            st.write(f"Femur Length in cm: {length_in_cm:.2f} cm")
            st.write(f"Fetus Height: {fetus_height_cm:.2f} cm")
            st.write(f"Gestational Age: {round(cal_gest(length_in_cm))} weeks")


# st.subheader("Preprocessing and Threshold Adjustment")
# with st.container():
#     threshold = st.slider("Threshold Value", min_value=0, max_value=255, value=125)
#     _, thresholded_img = cv2.threshold(filtered_image, threshold, 255, cv2.THRESH_BINARY)
#     st.image(thresholded_img, caption=f"Thresholded Image at {threshold}", width=512)

# # Process with selected threshold
# if st.button("Process with Selected Threshold"):
#     segmented_img = segment_image(thresholded_img)
#     contours, convex_hull_img = apply_convex_hull(segmented_img)
#     femur_length, point1, point2, marked_image = extract_femur_length(convex_hull_img, resized_image)
#     marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)

#     st.subheader("Segmented Image with Convex Hull")
#     with st.container():
#         st.image(convex_hull_img, caption="Convex Hull", width=512)

#     st.subheader("Marked Image with Femur Length")
#     with st.container():
#         st.image(marked_image, caption="Marked Image with Femur Length", channels="RGB", width=512)
#         st.write(f"Femur Length: {femur_length:.2f} pixels")

#     # Calculate and display fetal height if femur length is available
#     if femur_length:
#         fetus_height_cm, length_in_cm = fetus_height(femur_length)
#         st.header("Results")
#         col6, col7 = st.columns(2)
#         with col6:
#             st.write(f"Femur Length in cm: {length_in_cm:.2f} cm")
#             st.write(f"Fetus Height: {fetus_height_cm:.2f} cm")