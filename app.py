from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Similarity thresholds and weights
SIMILARITY_THRESHOLD = 0.7
PREFERENCE_WEIGHT = 0.7
PERSONALITY_WEIGHT = 0.3

# Helper: Cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Convert numpy.float32 to float
def convert_to_float(data):
    if isinstance(data, dict):
        return {k: convert_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_float(v) for v in data]
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        return float(data)
    return data

@app.route('/recommend', methods=['POST'])
def recommend_rooms():
    data = request.get_json()

    user_habits = data['user_habits']  # Dictionary
    user_personality = data['user_personality']  # List of float values
    agreements = data['agreements']  # List of agreements

    # Get SBERT embedding for user habits
    user_habit_text = " ".join(user_habits.values())
    user_habit_vector = model.encode(user_habit_text)

    recommendations = []

    for agreement in agreements:
        room_id = agreement['roomId']
        users = agreement['users']

        for other in users:
            other_user_id = other['userId']
            other_habits = other['habits']
            other_personality = other['personality']  # List of floats

            # Encode other user’s habits
            other_habit_text = " ".join(other_habits.values())
            other_habit_vector = model.encode(other_habit_text)

            # Compute similarities
            habit_similarity = cosine_similarity(user_habit_vector, other_habit_vector)
            personality_similarity = cosine_similarity(user_personality, other_personality)

            # Final score (weighted)
            total_similarity = (
                PREFERENCE_WEIGHT * habit_similarity +
                PERSONALITY_WEIGHT * personality_similarity
            )

            if total_similarity >= SIMILARITY_THRESHOLD:
                recommendations.append({
                    'roomId': room_id,
                    'userId': other_user_id,
                    'similarity': total_similarity,
                    'habit_similarity': habit_similarity,
                    'personality_similarity': personality_similarity,
                    'habits': other_habits
                })

    # Convert types and sort
    recommendations = convert_to_float(recommendations)
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
