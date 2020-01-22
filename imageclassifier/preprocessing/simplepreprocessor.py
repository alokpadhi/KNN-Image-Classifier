"""This module is to prepare basic preprocessing for images."""
import cv2

class SimplePreprocessor: # pylint: disable=too-few-public-methods
    """
    Preprocessor class
    """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height and interpolation
        self.width = width
        self.height = height
        self.inter = inter


    def preprocess(self, image):
        """
        returns resized image
        """
        # resize the image
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
