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

import pandas as pd
import os
from mlxtend.preprocessing import minmax_scaling

mp_face_mesh = mp.solutions.face_mesh

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

name = 'a-fer2013_neutral'

DESIRED_HEIGHT = 680
DESIRED_WIDTH = 680

total_points = 468
Matrix = np.zeros(shape=(total_points, 3))

def min_max_scaling(matrix):
    # Find the minimum and maximum values in the matrix
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)
    
    # Scale the matrix by subtracting the minimum and dividing by the range
    scaled_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    
    return scaled_matrix

img_path = glob.glob(f"data/fer2013/test/neutral/*.png")
print("img_path", img_path)

neutrallist = []

imgnumber = 0

def individual_neutral():
    for image_name in img_path:
        img = cv2.imread(image_name)

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
                
            i = 0
            while i < total_points:
                Matrix[i][0]=results.multi_face_landmarks[0].landmark[i].x
                Matrix[i][1]=results.multi_face_landmarks[0].landmark[i].y
                Matrix[i][2]=results.multi_face_landmarks[0].landmark[i].z
                i += 1

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
            neutrallist.append(Dist_Matrix_scaled)

individual_neutral()

c = sum(neutrallist)/len(neutrallist)
np.save(name, c)


