"""Loading the dataset"""
import os
import numpy as np
import cv2

class SimpleDatasetLoader: # pylint: disable=too-few-public-methods
    """Dataset Loader with preprocessing"""
    def __init__(self, preprocessors=None):
        # store the image preprocessors
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
    def load(self, imgpaths, verbose=-1):
        """Loading the images"""
        data = []
        labels = []

        for (i, imgpath) in enumerate(imgpaths):
            image = cv2.imread(imgpath)
            label = imgpath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)

            # showing update on processed images using verbose
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imgpaths)))
        return (np.array(data), np.array(labels))
