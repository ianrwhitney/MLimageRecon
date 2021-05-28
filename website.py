# load the operating system library
import os

# webiste libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename

# Load math library
import numpy as np

# load machine learning libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import tensorflow as tf

# My two categories
X = 'deer'
Y = 'geese'

# Two example images for the website, are in static dir
sampleX = 'static/deer.jpg'
sampleY = 'static/geese.jpg'

# Where user uploads are stored
UPLOAD_FOLDER = 'static/uploads'
# Allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create the website object
app = Flask(__name__)
SECRET_KEY = os.urandom(42)

# create a running list of results
results = []


def load_model_from_file():
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModel = load_model('saved_model.h5')
    myGraph = tf.compat.v1.get_default_graph()
    return mySession, myModel, myGraph


# Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Initial webpage load
    if request.method == 'GET':
        return render_template('index.html', myX=X, myY=Y, mySampleX=sampleX, mySampleY=sampleY)
    else:  # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type' + str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        # When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))


# Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER + "/" + filename, target_size=(150, 150))
    test_image = image.img_to_array(test_image)  # turns user supplied image to an array so it can be proccessed by ML algo
    test_image = np.expand_dims(test_image, axis=0)
    # test_image = tf.data.Dataset.from_tensor_slices(test_image)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    # with myGraph.as_default():
    # set_session(mySession)
    # prob_model = tf.keras.Sequential([myModel, tf.keras.layers.SeparableConv2D()])
    result = myModel.predict(test_image)  # asking the model to predict: result is a number between 0-1 if it is 0 we think it is category X and if it is 1 it is category Y

    print(result[0][0])
    image_src = "/" + UPLOAD_FOLDER + "/" + filename
    if result[0][0] < 0.5:
        answer = "<div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img-thumbnail' /><h4>guess:" + X + " " + str(
            result[0]) + "</h4></div><div class='col'></div><div class='w-100'></div>"
    else:
        answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img-thumbnail' /><h4>guess:" + Y + " " + str(
            result[0]) + "</h4></div><div class='w-100'></div>"
    results.append(answer)
    return render_template('index.html', myX=X, myY=Y, mySampleX=sampleX, mySampleY=sampleY, len=len(results),
                           results=results)


def main():
    # loads the h5 file ie the brain
    mySession, myModel, myGraph = load_model_from_file()
    app.config['SECRETE_KEY'] = SECRET_KEY
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

    app.run()


# Launch everything
main()
