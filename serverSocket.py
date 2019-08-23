
import cv2 as cv
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
from werkzeug.utils import secure_filename
from tqdm import tqdm_notebook as tqdm
from utilities import assign_next_frame





from flask import Flask, render_template, session, request, \
    copy_current_request_context, jsonify, g, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
from flask_expects_json import expects_json

import json
import base64
import logging
import time

from db import raise_complain as save_raise_complain, \
    connect as db_connect, save_register_noti
# from gevent import monkey
# monkey.patch_all()


# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.

async_mode = None
# async_mode = 'eventlet'
# async_mode = 'gevent'
no_of_client = 0
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload

CORS(app, supports_credentials=True)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    class_names = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']


# Load car detecting RCNN Model
# Create model object in inference mode.
config = InferenceConfig()
car_model = modellib.MaskRCNN(
    mode="inference", model_dir=os.getcwd(), config=config)
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
    check = '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in allowed_set
    return check


params = {
    'ping_timeout': 10,
    'ping_interval': 5
}

socketio = SocketIO(app, async_mode=async_mode, **params)


token = None

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload

cancel_call_repeatedly = None


complainSchema = {
    'type': 'object',
    'properties': {
        'against_user_id': {'type': 'number'},
        'parking_lot_id': {'type': 'number'},
        'parking_space_id': {'type': 'number'},
        'details': {'type': 'string'},
        'created_by': {'type': 'number'}
    },
    'required': ['against_user_id', 'details', 'created_by'],
}

registerNoti = {
    'type': 'object',
    'properties': {
        'user_id': {'type': 'string'},
        'token': {'type': 'string'},
    },
    'required': ['user_id', 'token'],
}


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': error.description}), 400)


@app.route('/raise-complain', methods=['POST'])
@expects_json(complainSchema)
def raiseComplain():
    data = request.get_json()
    # complain_id = save_raise_complain(
    #     data["against_user_id"],
    #     data.get("parking_lot_id", None),
    #     data.get("parking_space_id", None),
    #     data["details"],
    #     data["created_by"]
    # )
    complain_id = save_raise_complain(**data)
    if type(complain_id) == int:
        return jsonify({
            "message": "Successfully raised complain",
            "success": True,
            "data": complain_id
        })
    else:
        return jsonify({
            "message": "Unable to raised complain",
            "success": False,
        }), 500


@app.route('/register-for-noti', methods=['POST'])
@expects_json(registerNoti)
def registerNoti():
    data = request.get_json()
    register_id = save_register_noti(**data)
    if type(register_id) == int:
        return jsonify({
            "message": "Successfully register-for-noti",
            "success": True,
            "data": register_id
        })
    else:
        return jsonify({
            "message": "Unable to register-for-noti",
            "success": False,
        }), 500


@app.route('/predict-car-no', methods=['POST'])
def plate_predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet plate_model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if 'file' not in request.files:
        return jsonify({'error': "No selected file"}), 400

    file = request.files['file']
    filename = file.filename

    if filename == "":
        return jsonify({'error': "No selected file"}), 400

    if file and allowed_file(filename=filename, allowed_set=allowed_set):
        # Read image file as numpy array of RGB dimension
        frame = io.imread(fname=file, pilmode='RGB')
        # frame = io.imread(fname=file)

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(
            frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the plate_model
        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv.getTickFrequency())

        # Get scores and geometry
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = car_plate.decode(
            scores, geometry, confThreshold)
        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(
            boxes, confidences, confThreshold, nmsThreshold)
        rectangles = 1
        detected_texts = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
                if j == 3:
                    roi = frame[int(vertices[1][1]):int(vertices[3][1]), int(
                        vertices[0][0]):int(vertices[3][0])]
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
            return jsonify({"data": detected_texts})

        else:
            return jsonify({'data': None})

    else:
        return jsonify({'data': None})


@socketio.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})


@socketio.on('my_broadcast_event', namespace='/test')
def test_broadcast_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         broadcast=True)


@socketio.on('join', namespace='/test')
def join(message):
    join_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('leave', namespace='/test')
def leave(message):
    leave_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('close_room', namespace='/test')
def close(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response', {'data': 'Room ' + message['room'] + ' is closing.',
                         'count': session['receive_count']},
         room=message['room'])
    close_room(message['room'])


@socketio.on('my_room_event', namespace='/test')
def send_room_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         room=message['room'])


@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)


@socketio.on('my_ping', namespace='/test')
def ping_pong():
    emit('my_pong')


@socketio.on('connect', namespace='/test')
def test_connect():
    global no_of_client
    no_of_client += 1
    if no_of_client == 1:
        pass
    emit('my_response', {'data': 'Connected', 'count': 0})
    app.logger.info('===CONNECTED====NO_OF_CLIENT=======>%s',
                    str(no_of_client))


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    global no_of_client
    # global cancel_call_repeatedly
    no_of_client -= 1
    if no_of_client == 0:
        pass
    # print('Client disconnected', request.sid)
    app.logger.info('===CLIENT DISCONNECTED==%s==NO_OF_CLIENT=======>%s',
                    request.sid, str(no_of_client))


@socketio.on_error_default  # handles all namespaces without an explicit error handler
def default_error_handler(e):
    print(e)
    pass


if __name__ == '__main__':
    db_connect()

    # Load FaceNet plate_model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 320
    inpHeight = 320
    plate_model = "model/EAST.pb"

    # Load network
    net = cv.dnn.readNet(plate_model)

    # Create a new named window
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
