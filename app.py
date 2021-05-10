from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import os
app = Flask(__name__)

#Load model
pipe = load(open('model/pipeline.pkl', "rb"))

@app.route('/')
def index():
    return 'Previsão na rota "/predict"'

@app.route('/predict', methods=['POST'])
def post():
    data= request.get_json(force=True)
    X=np.asarray(list(data.values())).reshape(1,-1)
    value=pipe.predict(X)[0]
    return jsonify({"Valor":round(float(value),2)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)