import numpy as np
import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('flight.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    a = [6, 7, 8]
    b = [9, 10, 11]
    c = [12, 1, 2, 3]
    d = [4, 5]
    farr = [int(x) for x in request.form.values()]
    if farr[1] in a:
        farr.append(0)
    elif farr[1] in b:
        farr.append(1)
    elif farr[1] in c:
        farr.append(2)
    else:
        farr.append(3)
    final_features = np.array(farr, dtype='int64')
    print(final_features)
    prediction = model.predict([final_features])
    output = round(prediction[0])

    if output == 0:
        return render_template('index.html', prediction_text='No delay'.format(output))
    elif output == 1:
        return render_template('index.html',
                               prediction_text='Chance of departure delay'.format(output))
    elif output == 2:
        return render_template('index.html',
                               prediction_text='Chance of both departure and arrival delay'.format(
                                   output))
    elif output == 3:
        return render_template('index.html',
                               prediction_text='Chance of flight being diverted'.format(output))
    elif output == 4:
        return render_template('index.html',
                               prediction_text='Chance of cancelling the flight'.format(output))
    else:
        return render_template('index.html', prediction_text='output {}'.format(output))

if __name__ == "__main__":
    os.environ.setdefault('FLASK_DEBUG', 'development')
    app.run(debug=False)
