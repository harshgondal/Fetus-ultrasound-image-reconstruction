import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import wiener as scipy_wiener

image_dir = './dataset/images'

df = pd.read_csv('./dataset/image_label.csv')
femur = df[df['Plane']=='Fetal femur'].iloc[16:]

def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def load_images(directory):
    images = []
    for filename in femur['Image_name']:
        img_path = os.path.join(directory, filename + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if image is not None:
            images.append(image)
    return images

def resizeImage(img):
    target_size = (512, 512)
    old_size = img.shape[:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    img_resized = cv2.resize(img, new_size)
    
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img_padded

def gaussian_filter(image, kernel_size=7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def segment_image(image, threshold):
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image 
    
    _, segmented_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    return segmented_img

def apply_convex_hull(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull_img = np.zeros_like(image)

    if contours:
        longest_distance = 0
        longest_contour = None
        for contour in contours:
            for i in range(len(contour)):
                for j in range(i + 1, len(contour)):
                    dist = cv2.norm(contour[i][0] - contour[j][0])
                    if dist > longest_distance:
                        longest_distance = dist
                        longest_contour = contour

        if longest_contour is not None:
            hull = cv2.convexHull(longest_contour)
            cv2.drawContours(convex_hull_img, [hull], -1, 255, thickness=cv2.FILLED)
    return contours, convex_hull_img

def extract_femur_length(image, original_image):
    marked_image = original_image.copy()
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_farthest_dist = 0
    point1, point2 = None, None
    
    for contour in contours:
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                dist = cv2.norm(contour[i][0] - contour[j][0])
                if dist > max_farthest_dist:
                    max_farthest_dist = dist
                    point1, point2 = tuple(contour[i][0]), tuple(contour[j][0])

    if point1 and point2:
        cv2.circle(marked_image, point1, 8, (0, 255, 0), -1)
        cv2.circle(marked_image, point2, 8, (0, 0, 255), -1)
        cv2.line(marked_image, point1, point2, (255, 0, 0), 3)
    
    marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
    return max_farthest_dist, point1, point2, marked_image

def fetus_height(femur_length):
    image_dpi = 96
    femur_length_in_cm = ((femur_length/image_dpi)*2.54)
    fetus_height = 6.18 + 0.59*femur_length_in_cm*10
    return fetus_height, femur_length_in_cm

def nafsm_filter(image, threshold=25, window_size=3):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_filtered = image.copy()
    rows, cols = img_filtered.shape
    offset = window_size // 2  
    
    # Traverse the image
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            window = img_filtered[i-offset:i+offset+1, j-offset:j+offset+1].flatten()
            median_val = np.median(window)
            if abs(img_filtered[i, j] - median_val) > threshold:
                img_filtered[i, j] = median_val
    
    return img_filtered

def srad_filter(image, iterations=10, kappa=30, gamma=0.1):
    img = image.astype('float32')
    for _ in range(iterations):
        # gradients
        delta_north = np.roll(img, -1, axis=0) - img
        delta_south = np.roll(img, 1, axis=0) - img
        delta_east = np.roll(img, -1, axis=1) - img
        delta_west = np.roll(img, 1, axis=1) - img

        # diffusion coefficients based on gradients
        c_north = np.exp(-(delta_north/kappa)**2)
        c_south = np.exp(-(delta_south/kappa)**2)
        c_east = np.exp(-(delta_east/kappa)**2)
        c_west = np.exp(-(delta_west/kappa)**2)

        # apply diffusion
        img += gamma * (c_north * delta_north + c_south * delta_south +
                        c_east * delta_east + c_west * delta_west)
    return np.clip(img, 0, 255).astype('uint8')

def wiener_filter(image, kernel_size=7):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    filtered_image = scipy_wiener(image_blurred, (kernel_size, kernel_size))
    
    filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=255, neginf=0)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    ## Temp fix
    filtered_image[image_blurred <= 20] = 0
    
    return filtered_image

def adaptive_median_filter(image, kernel_size=3, max_kernel_size=7):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape

    padded_image = np.pad(image, ((max_kernel_size // 2, max_kernel_size // 2), 
                                  (max_kernel_size // 2, max_kernel_size // 2)), 
                          mode='constant', constant_values=0)

    img_filtered = np.copy(image)

    for i in range(rows):
        for j in range(cols):
            kernel_size = 3 
            while kernel_size <= max_kernel_size:
                half_size = kernel_size // 2
                window = padded_image[i:i+kernel_size, j:j+kernel_size]

                min_val = np.min(window)
                max_val = np.max(window)
                median_val = np.median(window)

                # adaptive median filter logic
                if min_val < median_val < max_val:
                    img_filtered[i, j] = median_val
                    break
                else:
                    kernel_size += 2

    return img_filtered

def hybrid_filter_p1(image):
    # NAFSM
    image = nafsm_filter(image)
    # SRAD
    image = srad_filter(image)
    return image

def cal_gest(femur_length):
    ga= 0.262**(2)*(femur_length) + 2*(femur_length)+11.5
    return ga

