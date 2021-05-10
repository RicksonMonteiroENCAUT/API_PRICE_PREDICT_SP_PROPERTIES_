from flask import Flask, request, jsonify
from joblib import load
import numpy as np
app = Flask(__name__)

#Load model
pipe = load(open('model/pipeline.pkl', "rb"))

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def post():
    data= request.get_json(force=True)
    X=np.asarray(list(data.values())).reshape(1,-1)
    value=pipe.predict(X)[0]
    return jsonify({"Valor":round(float(value),2)})

if __name__ == '__main__':
    app.run()