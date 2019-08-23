#!/usr/bin/env python3

import cv2 as cv
import math
import argparse
import sys
import os
import pytesseract
import car_plate
import coco
import pandas as pd
import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import matplotlib.pyplot as plt
from skimage import io
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tqdm import tqdm_notebook as tqdm
from utilities import assign_next_frame


app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload



class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    class_names = [ 'bicycle', 'car', 'motorcycle','bus','truck']


# Load car detecting RCNN Model
# Create model object in inference mode.
config = InferenceConfig()
car_model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)
# Load weights trained on MS-COCO
car_model.load_weights("model/RCNN.h5", by_name=True)


def allowed_file(filename, allowed_set):
    """Checks if filename extension is one of the allowed filename extensions for upload.

    Args:
        filename: filename of the uploaded file to be checked.
        allowed_set: set containing the valid image file extensions.

    Returns:
        check: boolean value representing if the file extension is in the allowed extension list.
                True = file is allowed
                False = file not allowed.
    """
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set
    return check


@app.route('/PlatePredictImage', methods=['POST', 'GET'])
def plate_predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet plate_model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            # Read image file as numpy array of RGB dimension
            frame = io.imread(fname=file, mode='RGB')

            # Get frame height and width
            height_ = frame.shape[0]
            width_ = frame.shape[1]
            rW = width_ / float(inpWidth)
            rH = height_ / float(inpHeight)

            # Create a 4D blob from frame.
            blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

            # Run the plate_model
            net.setInput(blob)
            output = net.forward(outputLayers)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

            # Get scores and geometry
            scores = output[0]
            geometry = output[1]
            [boxes, confidences] = car_plate.decode(scores, geometry, confThreshold)
            # Apply NMS
            indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
            rectangles = 1
            detected_texts=[]
            for i in indices:
                # get 4 corners of the rotated rect
                vertices = cv.boxPoints(boxes[i[0]])
                # scale the bounding box coordinates based on the respective ratios
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                    if j == 3:
                        roi = frame[int(vertices[1][1]):int(vertices[3][1]), int(vertices[0][0]):int(vertices[3][0])]
                        # cv.imshow('aft', roi)
                        config = (
                            '-l eng -c tessedit_char_whitelist=ABJDRSXTEGKLZNHUV0123456789 --oem 1 --psm 11')
                        # Run tesseract OCR on image
                        text = pytesseract.image_to_string(roi, config=config)
                        # Print recognized text
                        if text != "" and len(text) >= 7:
                            print("Text number " + str(rectangles) + ": " + text)
                            detected_texts.append(text)
                            # cv.waitKey()
                            #cv.rectangle(frame, (vertices[0][0], vertices[1][1]), (vertices[3][0], vertices[3][1]),(0, 255, 0), 3)
                            # vertices[0][1] = vertices[0][1] + 100
                            #cv.putText(frame, str(rectangles) + str(text), (vertices[0][0], vertices[0][1]),cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
                            rectangles += 1

            if detected_texts:
                # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                return render_template('car_plate_predict_result.html', identity=detected_texts)

            else:
                return render_template(
                    'car_plate_predict_result.html',
                    identity="No car plates detected! Please manually add it!"
                    )

        else:
                return render_template(
                    'car_plate_predict_result.html',
                    identity="Operation was unsuccessful! No car plates detected."
                )
    else:
        return "POST HTTP method required!"


@app.route('/ParkPredictImage', methods=['POST', 'GET'])
def park_predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet car_model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename
        uploaded_files = request.files.getlist('file')

        if filename == "":
            return "No selected files"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):

            # Read image file as numpy array of RGB dimension
            frame = io.imread(fname=file, mode='RGB')
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            # Run the image through the Mask R-CNN model to get results.
            results = car_model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # How many frames of video we've seen in a row with a parking space open
            free_space_frames = 0

            park_slots = get_car_boxes(r['rois'], r['class_ids'])
            # Creating pandas dataframe from numpy array park_slots[["y1", "x1", "y2", "x2"]].astype(int)
            car_park_slots = pd.DataFrame({'y1': park_slots[:, 0], 'x1': park_slots[:, 1] ,  'y2': park_slots[:, 2],  'x2': park_slots[:, 3]})
            #my_park_slots = park_slots[['x1', 'y1', 'x2', 'y2']].astype(int)

            car_park_slots.to_csv("./parkings/park.csv", index=False)
            # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
            total_rows = car_park_slots.count
            #total_rows = total_rows+1
            return render_template('parking_slot_predict_result.html', identity="We detected "+str(total_rows)+"car parking slots.")

        else:
                return render_template(
                    'parking_slot_predict_result.html',
                    identity="Operation was unsuccessful! No parking slots detected."
                )
    else:
        return "POST HTTP method required!"



# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


@app.route('/CarPredictImage', methods=['POST', 'GET'])
def car_predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet car_model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):


            # Read image file as numpy array of RGB dimension
            frame = io.imread(fname=file, mode='RGB')
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]


            # Run the image through the Mask R-CNN model to get results.
            results = car_model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # How many frames of video we've seen in a row with a parking space open
            free_space_frames = 0

            if os.path.exists('./parkings/park.csv'):
                park_slots = pd.read_csv("./parkings/park.csv")
                outbox = park_slots[["y1", "x1", "y2", "x2"]].astype(int)
                parked_car_boxes = outbox.to_numpy()
            else:
                parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])

            # Get where cars are currently located in the frame
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])

            # Assume no spaces are free until we find one that is free
            free_space = False
            available_spaces = 0
            occupied_spaces = 0

            if car_boxes.size > 0 and parked_car_boxes.size > 0:
                # See how much those cars overlap with the known parking spaces
                overlaps = utils.compute_overlaps(parked_car_boxes, car_boxes)

                # Loop through each known parking space box
                for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

                    # For this parking space, find the max amount it was covered by any
                    # car that was detected in our image (doesn't really matter which car)
                    max_IoU_overlap = np.max(overlap_areas)

                    # Get the top-left and bottom-right coordinates of the parking area
                    y1, x1, y2, x2 = parking_area

                    # Check if the parking space is occupied by seeing if any car overlaps
                    # it by more than 0.15 using IoU

                    if max_IoU_overlap < 0.15:
                        # Parking space not occupied! Draw a green box around it
                        available_spaces +=1
                        # Flag that we have seen at least one open space
                        free_space = True
                    else:
                        # Parking space is still occupied - draw a red box around it
                        occupied_spaces +=1

            modelresults="available spaces: "+str(available_spaces)+"\n"+"  occupied spaces: "+str(occupied_spaces)

            # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
            return render_template('car_occupancy_predict_result.html', identity=modelresults)

        else:
                return render_template(
                    'car_occupancy_predict_result.html',
                    identity="Operation was unsuccessful! Nothing detected."
                )
    else:
        return "POST HTTP method required!"


@app.route("/")
def index_page():
    """Renders the 'index.html' page for manual image file uploads."""
    return render_template("index.html")


@app.route("/plate_predict")
def plate_predict_page():
    """Renders the 'car_plate_predict.html' page for manual image file uploads for prediction."""
    return render_template("car_plate_predict.html")


@app.route("/park_predict")
def park_predict_page():
    """Renders the 'car_plate_predict.html' page for manual image file uploads for prediction."""
    return render_template("parking_slot_predict.html")

@app.route("/car_predict")
def car_predict_page():
    """Renders the 'car_plate_predict.html' page for manual image file uploads for prediction."""
    return render_template("car_occupancy_predict.html")



if __name__ == '__main__':

    # Load FaceNet plate_model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 320
    inpHeight = 320
    plate_model ="model/EAST.pb"


    # Load network
    net = cv.dnn.readNet(plate_model)

    # Create a new named window
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Start flask application on waitress WSGI server
    #serve(app=app, host='0.0.0.0', port=5000)
    app.run(debug=False,threaded = False)
