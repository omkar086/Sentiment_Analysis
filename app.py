from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Serve the index.html file when accessing the root URL
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the tweet from the request
        data = request.get_json()  # Get JSON data
        tweet = data.get('tweet', '')

        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400
        
        print(f"Received tweet: {tweet}")
        
        # Transform the tweet using the vectorizer
        transformed_tweet = vectorizer.transform([tweet])

        print(f"Transformed tweet shape: {transformed_tweet.shape}")

        # Get the sentiment prediction
        sentiment = model.predict(transformed_tweet)

        print(f"Model prediction: {sentiment}")

        # Convert the numerical prediction to sentiment
        predicted_value = sentiment[0]
        if sentiment == 1:
            sentiment_label = 'positive'
        elif sentiment == 0:
            sentiment_label = 'neutral'
        else:
            sentiment_label = 'negative'
        
        # Return the sentiment prediction
        return jsonify({'sentiment': sentiment_label})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
