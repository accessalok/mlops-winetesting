from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('./best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    if isinstance(json_data, list):
        data = pd.DataFrame(json_data)
    else:
        data = pd.DataFrame([json_data])
    
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
