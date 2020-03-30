# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import operator
import face_recognition

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

known_faces = {}
known_encs = []
known_names = []

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lastEncoding = []
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

def get_jetson_gstreamer_source(capture_width=1280, capture_height=960, display_width=1280, display_height=960, framerate=30, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
	    "tcambin serial=50910677 ! video/x-raw, format=BGRx,width= 1280,height=960, framerate=30/1 ! appsink", cv2.CAP_GSTREAMER
            )

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
vs = VideoStream(src=gstreamer_pipeline(flip_method=0)).start()
#vs = VideoStream(src=get_jetson_gstreamer_source().start()

time.sleep(0.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index_fr.html")

def detect_faces(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, known_faces, known_encs, lastEncoding

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # grab the current timestamp and draw it on the frame
        #timestamp = datetime.datetime.now()
		#cv2.putText(frame, timestamp.strftime(
		#	"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        enc = face_recognition.face_encodings(frame)
        face_locations = face_recognition.face_locations(frame)
        ix = 0
        for fl in face_locations:
            cv2.rectangle(frame, (fl[3], fl[0]), (fl[1], fl[2]), (0, 0, 255), 2)
            cv2.rectangle(frame, (fl[3]-1, fl[0]-22), (fl[1]+1, fl[0]), (0, 0, 255), -1)

            name = "Unknown"
            res = face_recognition.face_distance(known_encs, enc[ix]).tolist()
            if min(res) < 0.6:
                name = known_names[res.index(min(res))]

            print(list(zip(known_names, res)))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (fl[3]+3, fl[0]-5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            lastEncoding = enc[0].tolist()

            ix += 1
        if len(face_locations) == 0:
            lastEncoding = []

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
		
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                    continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/last_encoding")
def last_encoding():
    global lastEncoding

    if len(lastEncoding) == 0:
        return Response("[]", mimetype = "text/plain")

    # return the response generated along with the specific media
    # type (mime type)
    return Response(str(lastEncoding[0:5]) + " ... " + str(lastEncoding[-5:]), mimetype = "text/plain")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    for filename in os.listdir("references"):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            name = filename[0:filename.index('.')]
            img = face_recognition.load_image_file("./references/"+filename)
            enc = face_recognition.face_encodings(img)[0]
            known_faces[name] = enc
            print("Found:", name)
    known_names = sorted(known_faces.keys())
    for k in known_names:
        known_encs.append(known_faces[k])

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_faces, args=(
		args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
