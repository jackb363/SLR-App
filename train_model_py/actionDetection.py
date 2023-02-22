import numpy as np
import cv2
from tensorflow import lite
import mediapipe as mp
import util
# Frame for ID word
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# predicts signs using exported tflite model
def detect_tflite(tflite_file_path):
    # Holistic model
    mp_holistic = mp.solutions.holistic
    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.6
    actions = np.array(['hello', 'iloveyou', 'thanks'])

    # Load the TFLite model and allocate memory for the interpreter
    interpreter = lite.Interpreter(model_path=tflite_file_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = util.mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            util.draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = util.extract_keypoints(results)
            keypoints = keypoints.astype('float32')
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                # add dimension so sequence is (1, 30, 1662) not (30, 1662)
                interpreter.set_tensor(input_details[0]['index'], np.expand_dims(sequence, axis=0))
                interpreter.invoke()
                res = interpreter.get_tensor(output_details[0]['index'])[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

