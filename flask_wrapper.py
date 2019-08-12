'''
Flask Wrapper For SIM OCR
~~~~~~
To run this app, simply run the following code
in your terminal

$ export FLASK_APP=flask_wrapper.py
$ flask run
'''
import flask
import cv2
import numpy as np
import base64
import logging
from SIM_OCR import SIM_OCR

app = flask.Flask(__name__)


def readb64(encoded_data):
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        image_b64_str = flask.request.form["image_b64"]
    except Exception:
        return "failed"

    image = readb64(image_b64_str)
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
            with open("./{0}.jpg".format(iccid_str), "wb") as fh:
                fh.write(base64.decodestring(str.encode(image_b64_str)))

            logging.info("Saved ./{0}.jpg as dataset".format(iccid_str))
        else:
            logging.info("Get invalid ICCID {0}".format(iccid_str))
            return "rejected"
    except Exception:
        logging.error("Error collecting images")
    return "Done"


if __name__ == '__main__':
    instruction = '''
Nah, this is not the right way to launch me.
To start the SIM OCR service, simply run:

    $ export FLASK_APP=flask_wrapper.py && flask run

in your shell
'''
    print(instruction)
