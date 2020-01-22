"""KNN classifier for images"""
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from imageclassifier.datasets.simpledatasetloader import SimpleDatasetLoader
from imageclassifier.preprocessing.simplepreprocessor import SimplePreprocessor

# argument parser
AP = argparse.ArgumentParser()
AP.add_argument("-d", "--dataset", required=True, help="path to input dataset")
AP.add_argument("-k", "--neighbors", type=int, default=1, help="number of nearest neighbours for \
                classification")
AP.add_argument("-j", "--jobs", type=int, default=-1, help="number of distance for knn distance")
args = vars(AP.parse_args())

print("[INFO] loading images...")
imgpaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor and load the images to the disk
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imgpaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.25, random_state=42)
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainx, trainy)
print(classification_report(testy, model.predict(testx),target_names=le.classes_))