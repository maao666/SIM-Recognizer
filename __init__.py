"""
SIM Card Recognizer
~~~~~~~~~~~
Required:
$ pip3 install opencv-contrib-python
"""

import cv2
import glob
import numpy as np
import logging
import pytesseract
import random
from pprint import pprint
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


# noinspection PyUnreachableCode
class SIM_Image(object):
    _raw_image = None
    _image = None

    @staticmethod
    def _rotate_bound(image, angle):
        """
        This method derived from imutils as the background color after rotation
            needs to be white
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        rotated_image = cv2.warpAffine(
            image, M, (nW, nH), borderValue=(255, 255, 255))
        return rotated_image

    def _black_n_white(self):
        self._image = cv2.adaptiveThreshold(self._image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105,
                                            10)

    @staticmethod
    def _normalize(source, destination):
        logging.debug("Performing normalization")
        destination = cv2.normalize(
            source, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def _get_minimum_area_rectangle(image):
        logging.debug("Getting the minimum effective range")
        gray = cv2.bitwise_not(image)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        return cv2.minAreaRect(coords)

    def _rotation_correction(self, offset_angle=0):
        logging.debug("Performing rotational correction")
        angle = self._get_minimum_area_rectangle(self._image)[-1]
        self._image = self._rotate_bound(self._image, angle)
        if self._image.shape[0] > self._image.shape[1]:
            self._image = self._rotate_bound(self._image, 90)

    def show_current_image(self, *args):
        if len(args) == 0:
            cv2.imshow("Current Image", self._image)
        else:
            cv2.imshow("Current Image", args[0])
        cv2.waitKey(0)

    def __init__(self, image_path: str, load_mode=cv2.IMREAD_GRAYSCALE,
                 normalization=True):
        self._raw_image = cv2.imread(image_path, 1)
        self._image = cv2.imread(image_path, load_mode)

        if normalization:
            self._normalize(self._raw_image, self._image)
        self._image = cv2.GaussianBlur(
            self._image, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
        self._image = cv2.medianBlur(self._image, 5)
        self._black_n_white()
        # Need to be replaced
        # self._image = self._image[:, :int(self._image.shape[1] / 2)]
        self._rotation_correction(offset_angle=0)

    def _detect_edge(self):
        edged = cv2.Canny(self._image, 30, 200)

    def _standardize(self, text: str) -> str:
        text_list = text.splitlines()
        result = ''
        for text_iter in text_list:
            if text_iter.strip() != '':
                result = '{0}{1}\n'.format(result, text_iter)
        return result.strip()

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

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    @staticmethod
    def _remove_outliers_by_ratio(boxes, max_ratio=0.83, min_ratio=0.25):
        picked_boxes = []
        for box in boxes:
            if min_ratio < box[2] / box[3] < max_ratio:
                picked_boxes.append(box)
        return picked_boxes

    @staticmethod
    def get_overall_overlay_rect(boxes):
        boxes = np.asarray(boxes)
        # Basically the idea is that in order to select the upper-left rectangle,
        # There must be:
        # * 3 rects with SIMILAR x and LARGER y appending to same_col
        # * AND 4 rects with SIMILAR y and LARGER x appending to same_row
        for candidate in boxes:
            boxes_in_row = []
            boxes_in_col = []
            acceptable_x = (candidate[0] - candidate[2] * 2 / 3, candidate[0] + candidate[2] * 2 / 3)
            acceptable_y = (candidate[1] - candidate[3] / 2, candidate[1] + candidate[3] / 2)
            acceptable_w = (candidate[2] * 0.7, candidate[2] * 1.3)  # Obsolete
            acceptable_h = (candidate[3] * 0.7, candidate[3] * 1.3)
            for box in boxes:
                if box[1] > candidate[1]+candidate[3] and acceptable_x[0] < box[0] < acceptable_x[1] \
                        and acceptable_h[0] < box[3] < acceptable_h[1]:
                    boxes_in_col.append(box)
                if box[0] > candidate[0]+candidate[2] and acceptable_y[0] < box[1] < acceptable_y[1] \
                        and acceptable_h[0] < box[3] < acceptable_h[1]:
                    boxes_in_row.append(box)
            if len(boxes_in_col) >= 3 and len(boxes_in_row) >= 4:
                break
        return candidate

    @staticmethod
    def _remove_outlier_by_upper_left_rect(boxes, upper_left_rect):
        picked_boxes = []
        accepted_x_range = (upper_left_rect[0] - upper_left_rect[3]/3, upper_left_rect[0] + upper_left_rect[3] * 3)
        accepted_y_range = (upper_left_rect[1] - upper_left_rect[3] / 3, upper_left_rect[1] + upper_left_rect[3] * 4)
        for box in boxes:
            if accepted_x_range[0] < box[0] < accepted_x_range[1] and accepted_y_range[0] < box[1] < accepted_y_range[1]:
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
        if len(boxes) < 20:
            logging.error("Detection failure: less than 20 digits")
        if len(boxes) > 20:
            logging.error("Detection failure: redundant digits")

    def _mser_detection(self):
        mser = cv2.MSER_create()
        enlarged_image = cv2.resize(
            self._image, (self._image.shape[1] * 2, self._image.shape[0] * 2))
        regions = mser.detectRegions(enlarged_image)
        boxes = self.non_max_suppression_fast(regions[1])
        boxes = self._remove_outliers_by_ratio(boxes)
        enlarged_image = cv2.cvtColor(enlarged_image, cv2.COLOR_GRAY2RGB)
        marked_image = enlarged_image.copy()
        upper_left_box = self.get_overall_overlay_rect(boxes)
        boxes = self._remove_outlier_by_upper_left_rect(boxes, upper_left_box)
        self._failure_detection(boxes)
        if __debug__:
            x, y, w, h = upper_left_box
            cv2.rectangle(marked_image, (int(x - h/3), int(y - h/3)), (int(x + h*3.5), int(y + h*4.8)), (100, 0, 242), 3)
            # boxes = self._remove_outliers_by_size(boxes)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(marked_image, (x, y), (x + w, y + h), (50, 240, 50), 3)

            cv2.imshow('Image', marked_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        box_matrix = self._get_box_matrix(boxes)
        return enlarged_image, box_matrix

    def get_serial(self) -> str:
        enlarged_image, box_matrix = self._mser_detection()
        result = ''
        for row in box_matrix:
            for col in row:
                cv2.imwrite('./dataset/' + str(random.randrange(1000000000)) + '.bmp',
                            enlarged_image[col[1] - 3:col[1] + col[3] + 3, col[0] - 3:col[0] + col[2] + 3])
        return result


def _perfcheck():
    file_list = glob.glob("./source/*.jpg")

    for path in file_list:
        sim = SIM_Image(path)
        print(sim.get_serial())


if __name__ == '__main__':
    _perfcheck()
