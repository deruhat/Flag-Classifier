import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from tkinter import *
from tkinter import filedialog
from tkinter import font

from PIL import Image

# Load model
model = load_model('saved_model.h5')
opt = keras.optimizers.SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

# Function to choose flag and predict its class
def predict():
    root.filename = filedialog.askopenfilename(initialdir = "", title = "Select flag:")
    if root.filename :
        img = Image.open(root.filename)
        img = img.resize((20, 20))
        arr = np.array((img_to_array(img)),)
        arr = arr.reshape((1, 20, 20, 3))
        predictions = model.predict(arr)
        predictions = np.around(predictions,decimals = 2)
        plot(predictions)
        print(predictions)
    return

# Function for plotting result of classification
def plot(predictions):
    fig, ax = plt.subplots()
    ind = np.arange(1, 18)
    prediction = predictions.ravel()
    af = plt.bar(1, prediction[0])
    al = plt.bar(2, prediction[1])
    alg = plt.bar(3, prediction[2])
    an = plt.bar(4, prediction[3])
    ang = plt.bar(5, prediction[4])
    ant = plt.bar(6, prediction[5])
    ar = plt.bar(7, prediction[6])
    arm = plt.bar(8, prediction[7])
    au = plt.bar(9, prediction[8])
    aus = plt.bar(10, prediction[9])
    az = plt.bar(11, prediction[10])
    bh = plt.bar(12, prediction[11])
    bhr = plt.bar(13, prediction[12])
    ba = plt.bar(14, prediction[13])
    bar = plt.bar(15, prediction[14])
    be = plt.bar(16, prediction[15])
    bel = plt.bar(17, prediction[16])
    ax.set_xticks(ind)
    ax.set_xticklabels(['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua & Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belize'])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show(block = False)

# GUI
root = Tk()
root.geometry('500x200')
root.title("Flag Recognizer")
font = font.Font(family='tahoma', size='15')
var = StringVar()
label = Label( root, textvariable=var, relief=RAISED, bd='0' , height=5, font=font)
var.set("Choose a file to make a prediction.")
label.pack()
button = Button(root, text=u"Choose File", command=predict, width = 15, height = 5)
button.pack(side = "top", expand = True)
root.mainloop()