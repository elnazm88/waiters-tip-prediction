from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
#creating app
app=Flask(__name__)
#upload the pickle model
model=pickle.load(open('tips1.pkl', 'rb'))

@app.route('/')
def temp():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    #get the data
    d1 = request.form['a']
    d2 = request.form['b']
    d3 = request.form['c']
    d4 = request.form['d']
    d5 = request.form['e']
    arr = np.array([[d1, d2, d3, d4, d5]])
    pred = model.predict(arr)
    return render_template('home.html', prediction_text="the tip prediction is {}.".format(pred))

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=80)