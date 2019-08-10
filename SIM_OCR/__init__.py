"""
SIM Card ICCID Recognizer
~~~~~~~~~~~
# Basic Usage:
    >>> from SIM_OCR import SIM_OCR
    >>> sim = SIM_OCR('file_path.jpg')
    >>> print(sim.get_serial())

# Required:
    $ pip3 install opencv-contrib-python keras

# Getting Dataset:
    >>> from SIM_OCR import SIM_OCR
    >>> sim = SIM_OCR('file_path.jpg')
    >>> sim.save_dataset(path='dataset_folder/')
    You might want to set file_name_correspondence to
        true so that the file would be automatically
        tagged.
"""

import cv2
import numpy as np
import logging
import random
import os
from io import BytesIO
from pprint import pprint
import imageio
from PIL import Image

from keras.models import load_model
import skimage
import numpy as np

__author__ = "Ma, Jiaao"
__version__ = "1.0"

__all__ = ['SIM_OCR']

_CATALOGUE = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
              6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
              12: 'C', 13: 'D', 14: 'E', 15: 'F'}

_DEBUG = False

classifier = load_model('sim_iccid_model.hdf5')

# To make the model available for multiple threads
classifier._make_predict_function()


def _identify(image):
    width = 60
    height = 90
    channels = 3
    image = image.reshape(1, width, height, channels)

    catalogue = classifier.predict_classes(image)
    prediction_result = _CATALOGUE[catalogue[0]]
    return prediction_result

# noinspection PyUnreachableCode


class SIM_OCR(object):
    _raw_image = None
    _image = None
    _boxes = None
    _enlarged_image = None
    _normalized_image = None
    _has_failed = False

    @staticmethod
    def _rotate_bound(image, angle):
        """
        This method derived from imutils as the background color after rotation
            needs to be white
        """
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated_image = cv2.warpAffine(
            image, M, (nW, nH), borderValue=(255, 255, 255))
        return rotated_image

    def _adaptive_black_n_white(self, raw_image, block_size=205, c_const=50):
        logging.debug(
            "Converting image to black n white with adaptive threshold")
        self._image = cv2.adaptiveThreshold(raw_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block_size, c_const)
        if _DEBUG:
            self.show_image()

    @staticmethod
    def _normalize(source):
        logging.debug("Performing normalization")
        return cv2.normalize(
            source, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def _get_minimum_area_rectangle(image):
        logging.debug("Getting the minimum effective range")
        gray = cv2.bitwise_not(image)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        return cv2.minAreaRect(coords)

    def _correct_angle(self, offset_angle=0):
        logging.debug("Performing rotational correction")
        angle = self._get_minimum_area_rectangle(self._image)[-1]
        self._image = self._rotate_bound(self._image, angle)
        if self._image.shape[0] > self._image.shape[1]:
            self._image = self._rotate_bound(self._image, 90)

    def show_image(self, *args):
        if len(args) == 0:
            cv2.imshow("Current Image", self._image)
        else:
            cv2.imshow("Current Image", args[0])
        cv2.waitKey(0)

    def _adaptive_resize(self, image, scale_percentage=40):
        while image.shape[1] > 1500 or image.shape[0] > 960:
            width = int(image.shape[1] * scale_percentage / 100)
            height = int(image.shape[0] * scale_percentage / 100)
            dim = (width, height)
            # resize image
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

    '''
    Brief instruction on solving over-splitting issue
    ~~~~~~
    The idea is that, to change blur dadius until 20 boxes are
    detected.
    '''

    def _find_the_best_image(self):
        results = []
        if self._normalization:
            self._normalized_image = self._normalize(self._image)
        for radius in range(1, 25, 4):
            logging.info("Trying with radius {}".format(str(radius)))
            if self._blur and radius > 1:
                temp_image = cv2.GaussianBlur(
                    self._normalized_image, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
                temp_image = cv2.medianBlur(temp_image, radius)
            else:
                temp_image = self._normalized_image
            if self._black_n_white:
                for c in range(10, 64, 16):
                    self._adaptive_black_n_white(
                        temp_image, block_size=205, c_const=c)
                    if self._rotation_correction:
                        self._correct_angle(self._rotation_offset_angle)
                    try:
                        enlarged_image, boxes = self._mser_detection()
                    except Exception:
                        logging.info(
                            "MSER detection has encountered an internal error")
                        continue
                    if len(boxes) == 20:
                        self._boxes = boxes
                        self._enlarged_image = enlarged_image
                        return
                    results.append((enlarged_image, boxes))
            else:
                self._image = temp_image
        # So far the detection has almost failed
        # Returning the best result
        if len(results) > 0:
            results = sorted(results, key=lambda x: abs(20 - len(x[1])))
            self._enlarged_image, self._boxes = results[0][0], results[0][1]

    def __init__(self, image, load_mode=cv2.IMREAD_GRAYSCALE,
                 normalization=True, black_n_white=True, blur=True,
                 rotation_correction=True, rotation_offset_angle=0):
        if isinstance(image, str):
            self._image_path = image
            self._raw_image = cv2.imread(image, 1)
            self._image = cv2.imread(image, load_mode)
        elif isinstance(image, np.ndarray):
            self._raw_image = image
            self._image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            logging.error("Unknown argument type")

        self._image = self._adaptive_resize(self._image)
        self._normalization = normalization
        self._blur = blur
        self._black_n_white = black_n_white
        self._rotation_correction = rotation_correction
        self._rotation_offset_angle = rotation_offset_angle
        self._find_the_best_image()
        cv2.imwrite("./temp.jpg", self._raw_image)

    @staticmethod
    def non_max_suppression_fast(boxes, overlapThresh=0.1):
        """Source:
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        """
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + x1
        y2 = boxes[:, 3] + y1

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    @staticmethod
    def _remove_outliers_by_ratio(boxes, max_ratio=0.83, min_ratio=0.25):
        picked_boxes = []
        for box in boxes:
            if min_ratio < box[2] / box[3] < max_ratio:
                picked_boxes.append(box)
        return picked_boxes

    @staticmethod
    def _get_overall_overlay_rect(boxes):
        boxes = np.asarray(boxes)
        # Basically the idea is that in order to select the upper-left rectangle,
        # There must be:
        # * 3 rects with SIMILAR x and LARGER y appending to same_col
        # * AND 4 rects with SIMILAR y and LARGER x appending to same_row
        if len(boxes) == 0:
            logging.info("No candidate box found!")
            return

        for candidate in boxes:
            boxes_in_row = []
            boxes_in_col = []

            acceptable_x = (candidate[0] - candidate[2]
                            * 2 / 3, candidate[0] + candidate[2] * 2 / 3)
            acceptable_y = (candidate[1] - candidate[3] / 2,
                            candidate[1] + candidate[3] / 2)
            acceptable_h = (candidate[3] * 0.7, candidate[3] * 1.3)

            for box in boxes:
                if box[1] > candidate[1] + candidate[3] and acceptable_x[0] < box[0] < acceptable_x[1] \
                        and acceptable_h[0] < box[3] < acceptable_h[1]:
                    boxes_in_col.append(box)
                if box[0] > candidate[0] + candidate[2] and acceptable_y[0] < box[1] < acceptable_y[1] \
                        and acceptable_h[0] < box[3] < acceptable_h[1]:
                    boxes_in_row.append(box)

            if len(boxes_in_col) >= 3 and len(boxes_in_row) >= 4:
                break
        return candidate

    @staticmethod
    def _remove_outlier_by_upper_left_rect(boxes, upper_left_rect):
        picked_boxes = []
        accepted_x_range = (
            upper_left_rect[0] - upper_left_rect[3] / 3, upper_left_rect[0] + upper_left_rect[3] * 3.5)
        accepted_y_range = (
            upper_left_rect[1] - upper_left_rect[3] / 3, upper_left_rect[1] + upper_left_rect[3] * 4)
        for box in boxes:
            if accepted_x_range[0] < box[0] < accepted_x_range[1] and accepted_y_range[0] < box[1] < accepted_y_range[
                    1]:
                picked_boxes.append(box)
        return picked_boxes

    @staticmethod
    def _remove_outliers_by_size(boxes, v_tolerance=0.3, h_tolerance=0.6):
        picked_boxes = []
        medians = np.median(boxes, axis=0)
        # means = np.median(boxes, axis=0)
        # standard_deviations = np.std(boxes, axis=0)

        for box in boxes:
            if medians[2] * (1 - h_tolerance) < box[2] < medians[2] * (1 + h_tolerance) \
                    and medians[3] * (1 - v_tolerance) < box[3] < medians[3] * (1 + v_tolerance):
                picked_boxes.append(box)
            '''
            if box[2] / box[3] < 1 / 1.2 \
                    and abs(box[2] - medians[2]) < factor * standard_deviations[2] \
                    and abs(box[3] - medians[3]) < factor * standard_deviations[3] \
                    and abs(box[0] - medians[0]) < factor * 0.5 * standard_deviations[0] \
                    and abs(box[1] - medians[1]) < factor * standard_deviations[1]:
                picked_boxes.append(box)
            '''

        return picked_boxes
        '''
        def is_in_range(arg1: int, arg2: int, tolerance: int) -> bool:
            if arg2 - tolerance < arg1 < arg2 + tolerance:
                return True
            return False
        boxes = list(boxes)
        boxes.sort(key=lambda x: x[0])
        picked_boxes = []
        picked_boxes_r2 = []
        for i in range(len(boxes)-3):
            if is_in_range(boxes[i][0], boxes[i + 1][0], horizontal_tolerance) and \
                    is_in_range(boxes[i][0], boxes[i + 2][0], horizontal_tolerance) and \
                    is_in_range(boxes[i][0], boxes[i + 3][0], horizontal_tolerance):
                picked_boxes.append(boxes[i])
        picked_boxes.sort(key=lambda x: x[1])
        for i in range(len(picked_boxes)-3):
            if is_in_range(picked_boxes[i][1], picked_boxes[i + 1][1], vertical_tolerance) and \
                    is_in_range(picked_boxes[i][1], picked_boxes[i + 2][1], vertical_tolerance) and \
                    is_in_range(picked_boxes[i][1], picked_boxes[i + 3][1], vertical_tolerance):
                picked_boxes_r2.append(picked_boxes[i])
        return np.asarray(picked_boxes_r2, dtype=np.int32)
        '''

    @staticmethod
    def _get_box_matrix(boxes, N=5):
        boxes = list(boxes)
        boxes.sort(key=lambda x: x[1])
        boxes = np.array(boxes)
        box_matrix = [boxes[n:n + N] for n in range(0, len(boxes), N)]
        for row in box_matrix:
            row.sort(axis=0)
        return box_matrix

    def _failure_detection(self, boxes):
        box_num = len(boxes)
        if box_num != 20:
            logging.warning(
                "Result might be inaccurate: {0} detected".format(box_num))
            return True
        else:
            logging.info("Standby.")
            return False

    def _mser_detection(self):
        mser = cv2.MSER_create()

        enlarged_image = cv2.resize(
            self._image, (self._image.shape[1] * 2, self._image.shape[0] * 2))
        enlarged_image_raw = self._adaptive_resize(self._raw_image)
        enlarged_image_raw = cv2.resize(
            enlarged_image_raw, (enlarged_image_raw.shape[1] * 2, enlarged_image_raw.shape[0] * 2))

        regions = mser.detectRegions(enlarged_image)

        boxes = self.non_max_suppression_fast(regions[1])
        boxes = self._remove_outliers_by_ratio(boxes)

        enlarged_image = cv2.cvtColor(enlarged_image, cv2.COLOR_GRAY2RGB)
        marked_image = enlarged_image_raw.copy()
        upper_left_box = self._get_overall_overlay_rect(boxes)
        boxes = self._remove_outlier_by_upper_left_rect(boxes, upper_left_box)

        if _DEBUG:
            x, y, w, h = upper_left_box
            x1, y1, x2, y2 = int(x - h / 3), int(y - h /
                                                 3), int(x + h * 4), int(y + h * 4.8)
            # enlarged_image = cv2.bitwise_not(enlarged_image)
            marked_image[y1:y2, x1:x2] = enlarged_image[y1:y2, x1:x2]

            cv2.rectangle(marked_image, (x1, y1), (x2, y2), (36, 253, 153), 2)
            # boxes = self._remove_outliers_by_size(boxes)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(marked_image, (x, y),
                              (x + w, y + h), (178, 100, 250), 2)

            cv2.imshow('Labeled Image', marked_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return enlarged_image, boxes

    def get_serial(self, offset=3, strict_mode=True) -> str:
        """
        Get ICCID
        ~~~~~~
        - returns a string containing the ICCID of the sim card

        - offset means how many additional pixels should be cropped
        outside each symbol rectangle

        - Setting strict_mode to True would return "failed" once
        less than 20 boxes are detected
        """
        if strict_mode and self._failure_detection(self._boxes):
            return "failed"

        try:
            box_matrix = self._get_box_matrix(self._boxes)
            result = ''
            for row in box_matrix:
                for box in row:
                    image = self._enlarged_image[box[1] - offset:box[1] + box[3] + offset,
                                                 box[0] - offset:box[0] + box[2] + offset]
                    # Perform recognition
                    image = cv2.resize(image, (90, 60))  # TODO
                    result = result + _identify(image)
        except Exception as e:
            print(e)
            return "failed"
        return result

    def save_dataset(self, path='./dataset/', file_format='bmp', offset=3,
                     strict_mode=True, file_name_correspondence=False):
        """
        Extract digits and save to dataset
        ~~~~~~
        - Extracts each digit to a folder containing the dataset
        The folder will be created if it does not exist

        - offset means how many additional pixels should be cropped
        outside each symbol rectangle

        - strict_mode means the dataset will be saved if and only if
        there are exactly 20 digits being recognized to 
        improve accuracy

        - file_name_correspondence can be enabled when the file
        name is exactly the ICCID such as 
        '898600B5151770173671.jpg', where all letters should
        be in upper-case. This is helpful for tagging images.
        Random file names would be used otherwise.
        """
        try:
            if strict_mode and len(self._boxes) != 20:
                logging.info("Skipping potentially inaccurate result")
                return
        except TypeError:
            logging.info("Skipping potentially invalid result")
            return

        box_matrix = self._get_box_matrix(self._boxes)
        sorted_boxes = []
        for row in box_matrix:
            for box in row:
                sorted_boxes.append(box)
        for box_index in range(len(sorted_boxes)):
            if len(path) >= 1 and path[-1] != '/':
                path = path + '/'
            if file_name_correspondence:
                image_name = self._image_path[self._image_path.rfind('/') + 1:]
                saving_path = '{}{}'.format(path, image_name[box_index])
                os.makedirs(saving_path, exist_ok=True)
                saving_name = '{}/{}.{}'.format(saving_path,
                                                str(random.randrange(100000000)), file_format)
            else:
                saving_name = '{}{}.{}'.format(path, str(random.randrange(100000000)),
                                               file_format)
            box = sorted_boxes[box_index]
            cv2.imwrite(saving_name,
                        self._enlarged_image[box[1] - offset:box[1] + box[3] + offset,
                                             box[0] - offset:box[0] + box[2] + offset])


if __name__ == '__main__':
    print("Refer demo.py to get started.")
    print("See documentation for more details.")
