from flask import Flask, jsonify, request, render_template
import model

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    prediction = model.predict(data['input'])
    response = {'prediction': prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)