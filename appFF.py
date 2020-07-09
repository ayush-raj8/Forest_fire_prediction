#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:49:32 2020

@author: ayushraj
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

appFF = Flask(__name__)
model = pickle.load(open('modelFF.pkl', 'rb'))

@appFF.route('/')
def home():
    return render_template('form2.html')

@appFF.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)
    output ='{0:.{1}f}'.format(prediction[0][1],2)

    if output>str(0.5):
        return render_template('form2.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="Take precautionary steps")
    else:
        return render_template('form2.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
if __name__ == "__main__":
    appFF.run(debug=True)
