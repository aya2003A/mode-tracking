from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import pickle
import uuid 
import os

from config import Config

app = Flask(__name__)

app.config.from_object(Config)
client = MongoClient(app.config['MONGO_URI'])

db = client['users']
users_collection = db['users']

journal_db = client['Journal']
journal_collection = journal_db['journal']

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Load the model
try:
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
except FileNotFoundError:
    print("xgb_model.pkl not found. Make sure the file is in the correct directory.")

# Label mapping
label_mapping = {
    0: 'Normal',
    1: 'Depression',
    2: 'Suicidal',
    3: 'Anxiety',
    4: 'Bipolar',
    5: 'Stress',
    6: 'Personality disorder'
}

# Preprocessing function
def preprocess_and_get_embeddings(text):
    preprocessed_text = bert_preprocess([text])
    embeddings = bert_encoder(preprocessed_text)['pooled_output']
    return embeddings.numpy()  

# Prediction function
def predict_new_sentence(sentence):
    embeddings = preprocess_and_get_embeddings(sentence)
    prediction = xgb_model.predict(embeddings)
    predicted_label = label_mapping.get(prediction[0], "Unknown")
    return predicted_label

# Mode tracking endpoint
@app.route("/api/mode_tracking", methods=["POST"])
def mode_track():
    data = request.get_json()
    if not data:
        return jsonify({"Alert": "You didn't write anything!"}), 400

    # Get email and sentence from request data
    email = data.get('email')
    title = data.get("title")
    sentence = data.get('sentence')
    
    if not email or not sentence:
        return jsonify({"Error": "'email', 'sentence' and 'title' are required fields."}), 400

    user_exists = users_collection.find_one({"email": email})
    if not user_exists:
        return jsonify({"Error": "Email not found"}), 404
    
    mode_track = predict_new_sentence(sentence)
    
    entry_id = str(uuid.uuid4())
    
    # Prepare journal entry data
    journal_entry = {
        "_id": entry_id,
        "title": title,
        "content": sentence,
        "prediction": mode_track
    }

    
    current_date = datetime.utcnow().strftime("%d-%m-%Y")

    user_journal = journal_collection.find_one({"email": email})
    
    if user_journal:
        # Check if there's already an entry for the current date
        date_entry = next((entry for entry in user_journal['journal'] if entry['date'] == current_date), None)
        
        if date_entry:
            # If date exists, append the new entry to 'entries'
            journal_collection.update_one(
                {"email": email, "journal.date": current_date},
                {"$push": {"journal.$.entries": journal_entry}}
            )
        else:
            # If date does not exist, create a new date entry with the journal entry
            new_date_entry = {
                "date": current_date,
                "entries": [journal_entry]
            }
            journal_collection.update_one(
                {"email": email},
                {"$push": {"journal": new_date_entry}}
            )
    else:
        
        new_journal_entry = {
            "email": email,
            "journal": [
                {
                    "date": current_date,
                    "entries": [journal_entry]
                }
            ]
        }
        journal_collection.insert_one(new_journal_entry)

    # Update the user's current mode in the users collection
    users_collection.update_one(
        {'email': email},
        {'$set': {'current_mode': mode_track}}
    )

    return jsonify({
        "message": "Journal entry analyzed successfully.",
        "content": sentence,
        "prediction": mode_track
    }), 200


if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    app.run(debug=True)
