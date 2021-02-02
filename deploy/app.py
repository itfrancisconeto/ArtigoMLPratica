import numpy as np
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
modelo = load('previsor.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/model/', methods=['GET', 'POST'])
def model():
    y_predict = 0
    try:        
        if request.method == 'POST':
            preditor = request.form.get("preditor")
            X_preditor = np.array([[preditor]]).astype(np.float64)
            y_predict = modelo.predict(X_preditor)
            y_predict = np.round(y_predict, 2)    
    except Exception as e:
        y_predict = 'Predict fail'
    return render_template('index.html', predict=y_predict)

if __name__ == "__main__":
    app.run(debug=False)