'''
WSGI Server For SIM OCR
~~~~~~
# Routes:

- 'recognize' receives a POST request containing base64
    encoding of the image and returns an ICCID

- 'collect' receives a POST containing both base64 and
    the ICCID of the image, writting an image file
    to disk. The path depends on different OS
'''
from os import urandom
from gevent.pywsgi import WSGIServer
from gevent import monkey
monkey.patch_all()  # noqa: E702
import flask
import cv2
import numpy as np
import base64
import logging
from SIM_OCR import SIM_OCR
import sys

app = flask.Flask(__name__)
app.secret_key = urandom(24)

LISTEN_TO = ('0.0.0.0', 1012)


def convert_base64(encoded_base64: str):
    numpy_array = np.fromstring(base64.b64decode(encoded_base64), np.uint8)
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return img


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        image_b64_str = flask.request.form["image_b64"]
    except Exception:
        return "failed"

    image = convert_base64(image_b64_str)
    sim = SIM_OCR(image, rotation_correction=False)
    serial = sim.get_serial(strict_mode=True)

    logging.info("Returning serial from remote image: " + serial)

    return serial


@app.route('/collect', methods=['POST'])
def collect():
    try:
        image_b64_str = flask.request.form["image_b64"]
        iccid_str = flask.request.form["ICCID"].upper()

        if len(iccid_str) == 20:
            if sys.platform == "linux" or sys.platform == "linux2":
                directory = "/collection"
            else:
                directory = "./collection"
            with open("{0}/{1}.jpg".format(directory, iccid_str), "wb") as fh:
                fh.write(base64.decodestring(str.encode(image_b64_str)))

            logging.info("Saved ./{0}.jpg as dataset".format(iccid_str))
        else:
            logging.info("Get invalid ICCID {0}".format(iccid_str))
            return "rejected"
    except Exception:
        logging.error("Error collecting images")
    return "Done"


if __name__ == '__main__':
    logging.info("Starting WSGI Server")
    http_server = WSGIServer(LISTEN_TO, app)
    http_server.serve_forever()
