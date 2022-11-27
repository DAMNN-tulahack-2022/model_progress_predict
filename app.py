from flask import Flask, request, jsonify
from model import trainAndPredict, dataToDataFrame

app = Flask(__name__)

@app.route('/')
def main():
    return "Hello in YOUR Flask REST-API!"

@app.route('/users/predict', methods=['GET'])
def getPredictById():
    data = request.json
    experience = data['experienceDynamic']
    authDate = data['authDate']

    user_prediction = list(trainAndPredict(dataToDataFrame(experience, authDate)))

    return jsonify({'predictionCurrentWeek': user_prediction})