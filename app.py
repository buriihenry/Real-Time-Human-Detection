import cv2
import imutils
from flask import Flask, render_template

# initialize the flask app
app = Flask(__name__)

# Intitilize the log desciptor and set SVM to pre-trained pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Route the app to the home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")