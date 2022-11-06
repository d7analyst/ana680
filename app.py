from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_HW2.pkl'
model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)  # two ways to load the model, not using joblib here
#model = joblib.load('filename.pkl')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    Clump_Thickness = request.form['Clump_Thickness']
    Uniformity_of_Cell_Size = request.form['Uniformity_of_Cell_Size']
    Marginal_Adhesion = request.form['Marginal_Adhesion']
    pred = model.predict(np.array([[Clump_Thickness, Uniformity_of_Cell_Size, Marginal_Adhesion]]))
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
