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

# route the app to the image detection page
@app.route("/image")
def image():
    image = cv2.imread("image.jpg")
    image = imutils.resize(image, width=min(500, image.shape[1]))
    regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale = 1.05)
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, f"Total Persons: {len(regions)}", (20, 315), cv2.FONT_ITALIC, 0.5, (0, 0,255), 1)
    cv2.imshow("Human Detection from an Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render_template("image.html")   