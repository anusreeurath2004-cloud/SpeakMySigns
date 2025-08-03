from flask import Flask, render_template, request, jsonify, send_file
import base64
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tensorflow.keras.models import load_model
import traceback
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pyttsx3
import os
import uuid

# Flask setup
app = Flask(__name__)

# Create a directory for audio files if it doesn't exist
os.makedirs('tts_output', exist_ok=True)

# Initialize pyttsx3 engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Setting the speaking rate

# Load gesture model
model = load_model('fullset.h5')
class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'Alright', 'Animal', 'B', 'Beautiful', 'Bed', 'Bedroom', 'Bird', 'Black', 'Blind',
    'C', 'Cat', 'Chair', 'Colour', 'Cow', 'D', 'Daughter', 'Deaf', 'Dog', 'Door', 'Dream',
    'E', 'F', 'Father', 'Fish', 'Friday', 'G', 'Good Morning', 'Good night', 'Grey',
    'H', 'Happy', 'He', 'Hello', 'Horse', 'How are you', 'I','Ii', 'It',
    'J', 'K', 'L', 'Loud', 'M', 'Monday', 'Mother', 'Mouse',
    'N', 'O', 'Orange', 'P', 'Parent', 'Pink', 'Pleased',
    'Q', 'Quiet', 'R', 'S', 'Sad', 'Saturday', 'She', 'Son', 'Sunday',
    'T', 'Table', 'Thank you', 'Thursday', 'Today', 'Tuesday',
    'U', 'Ugly', 'V', 'W', 'Wednesday', 'White', 'Window',
    'X', 'Y', 'You', 'Z'
]  # your full class list here, same as before

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load sentence generation model at startup
print("Loading T5 model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained('./flan-t5-customm')
sentence_model = T5ForConditionalGeneration.from_pretrained(
    './flan-t5-customm',
    device_map='auto',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
print("T5 model loaded successfully!")

def extract_keypoints(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(results.multi_hand_landmarks) == 1:
            keypoints.extend([0] * 63)
    else:
        keypoints = [0] * 126
    return keypoints, results

def generate_speech(text):
    # Generate a unique filename
    filename = f"tts_output/{str(uuid.uuid4())}.wav"
    
    # Generate speech using pyttsx3
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    
    return filename

def generate_sentence_from_words(words):
    prompt = (
        "form a valid and grammatically correct sentence using the following words only once with proper structure and verb form: "
        + ", ".join(words)
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move input to the same device as the model's first parameter
    device = next(sentence_model.parameters()).device
    input_ids = input_ids.to(device)
    
    outputs = sentence_model.generate(
        input_ids=input_ids,
        max_length=30,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/')
def index():
    cleanup_old_files()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        frames = data.get("frames", [])

        if len(frames) != 30:
            return jsonify({"error": "Expected 30 frames"}), 400

        sequence = []
        for frame_data in frames:
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            keypoints, _ = extract_keypoints(frame)
            sequence.append(keypoints)

        time_indices = np.linspace(0, 1, 30).reshape(30, 1)
        sequence_with_time = np.concatenate([sequence, time_indices], axis=1)
        input_seq = np.expand_dims(sequence_with_time, axis=0)

        prediction = model.predict(input_seq)[0]
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    try:
        data = request.json
        words = data.get("words", [])

        if not words or not isinstance(words, list):
            return jsonify({"error": "Invalid word list"}), 400

        # Generate sentence
        sentence = generate_sentence_from_words(words)
        
        # Generate speech from the sentence
        audio_file = generate_speech(sentence)
        
        return jsonify({
            "sentence": sentence,
            "audio_url": f"/get_audio/{os.path.basename(audio_file)}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_audio/<filename>')
def get_audio(filename):
    try:
        return send_file(
            f"tts_output/{filename}",
            mimetype="audio/wav",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# Add cleanup function to remove old audio files
def cleanup_old_files():
    import glob
    import time
    
    # Delete files older than 1 hour
    current_time = time.time()
    for file in glob.glob("tts_output/*.wav"):
        if os.path.getmtime(file) < current_time - 3600:  # 3600 seconds = 1 hour
            try:
                os.remove(file)
            except:
                pass

if __name__ == '__main__':
    try:
        # Set host to '0.0.0.0' to make it accessible from other devices in the network
        # You can change the port if 5000 is already in use
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        
    finally:
        # Cleanup on exit
        print("Cleaning up...")
        cleanup_old_files()
        # Release pyttsx3 engine
        tts_engine.stop()
