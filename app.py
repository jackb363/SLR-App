from flask import Flask, render_template, request, jsonify
from tensorflow import lite
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)

interpreter = lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def detect_tflite():
    data = request.get_json()
    actions = np.array(['hello', 'iloveyou', 'thanks'])

    sequence = np.array(data['data']).astype('float32').reshape(30, 1662)
    # add dimension so sequence is (1, 30, 1662) not (30, 1662)
    sequence = np.expand_dims(sequence, axis=0)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]
    header = str(actions[np.argmax(res)])
    print(header)
    return jsonify({'result': header})


if __name__ == "__main__":
    app.run(debug=True)
