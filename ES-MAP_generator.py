import math
import numpy as np
import mediapipe as mp
import glob
import json
import cv2
import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colorbar as cb
from matplotlib import colors
from scipy import io
import scipy.io
import time

import pandas as pd
import os
from mlxtend.preprocessing import minmax_scaling

# Emotions for dataset with contempt
emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# Set upper and lower thresholds
upper_threshold = .1
lower_threshold = -.1


name = f'data/ES-NET L{lower_threshold} U{upper_threshold}'

os.makedirs(name, exist_ok=True)

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        normalized_min = max(0, 1/ 2* (1- abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1/ 2* (1+ abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def min_max_scaling(matrix):
    # Find the minimum and maximum values in the matrix
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)
    
    # Scale the matrix by subtracting the minimum and dividing by the range
    scaled_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    
    return scaled_matrix

# for each emotion listed in motion (labels)
for emotion_name in emotion:
    emotion_dir = os.path.join(name, emotion_name)
    os.makedirs(emotion_dir, exist_ok=True)

    emotion_name1 = emotion_name
    emotionlist = []

    # load neutral baseline
    mat = np.load('a-fer2013_neutral.npy') 
    mat = mat.reshape((468, 468))

    mp_face_mesh = mp.solutions.face_mesh

    # Load drawing_utils and drawing_styles
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    total_points = 468
    Matrix = np.zeros(shape=(total_points, 3))
    
    img_path = glob.glob(f"dataset/fer2013/train/{emotion_name1}/*.png")

    imgnumber = 0

    # for each image in the labeled folder
    for image_name in img_path:    
        imgnumber += 1
        img = cv2.imread(image_name)
        full_name = os.path.basename(image_name)
        file_name = os.path.splitext(full_name)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face landmarks of each face.
            if not results.multi_face_landmarks:
                continue
            annotated_image = img.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

                
            ###############################################################################################
        
            i = 0

            while i < total_points:

                Matrix[i][0]=results.multi_face_landmarks[0].landmark[i].x
                Matrix[i][1]=results.multi_face_landmarks[0].landmark[i].y
                Matrix[i][2]=results.multi_face_landmarks[0].landmark[i].z

                i += 1

            ###############################################################################################
           
            Dist_Matrix = np.zeros(shape=(total_points, total_points))

            b=0

            while b < total_points:

                a=0

                while a < total_points:

                    delta_x_squared = math.pow((Matrix[a][0]-Matrix[b][0]), 2)
                    delta_y_squared = math.pow((Matrix[a][1]-Matrix[b][1]), 2)
                    delta_z_squared = math.pow((Matrix[a][2]-Matrix[b][2]), 2)

                    Dist_Matrix[a][b]= math.sqrt(delta_x_squared + delta_y_squared + delta_z_squared)

                    a += 1
                b += 1

            Dist_Matrix_scaled = min_max_scaling(Dist_Matrix)
     
            # subtracting the baseline        

            baselinesubstacted = Dist_Matrix_scaled-mat

            baselinesubstacted[(baselinesubstacted > 0) & (baselinesubstacted < lower_threshold)] = 0
            baselinesubstacted[(baselinesubstacted < 0) & (baselinesubstacted > -lower_threshold)] = 0

            baselinesubstacted[(baselinesubstacted > 0) & (baselinesubstacted > upper_threshold)] = upper_threshold
            baselinesubstacted[(baselinesubstacted < 0) & (baselinesubstacted < -upper_threshold)] = -upper_threshold
           
            # Creating the plot

            vmin = -upper_threshold
            vmax = upper_threshold

            fig, ax = plt.subplots(figsize=(1, 1))
            fig.subplots_adjust(bottom=0.25)
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint = 0)
            plt.imshow(baselinesubstacted, cmap = "RdBu_r", interpolation='nearest', norm = norm)

            plt.savefig(f'data/{name}/{emotion_name1}/'+f'{emotion_name1}{imgnumber}.png')
            plt.close('all')