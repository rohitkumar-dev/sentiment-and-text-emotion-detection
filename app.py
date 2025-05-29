from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# Define the models and tokenizers using online loading
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"

# Load sentiment model and tokenizer
tokenizer_sentiment = AutoTokenizer.from_pretrained(sentiment_model_name)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load emotion model and tokenizer using pipeline for convenience
emotion_classifier = pipeline('sentiment-analysis', model=emotion_model_name)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home_page():
    return render_template("home.html")

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment_page():
    return render_template("sentiment.html")

@app.route("/sentiment_result", methods=["GET", "POST"])
def sentiment_result_page():
    labels = ['Negative', 'Neutral', 'Positive']
    input_sentiment = request.form['input_sentiment']
    input_sentiment_words = []
    for word in input_sentiment.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        input_sentiment_words.append(word)
    input_proc = " ".join(input_sentiment_words)
    encoded_tweet = tokenizer_sentiment(input_proc, return_tensors='pt')
    output = model_sentiment(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    result_data = []
    for i in range(len(scores)):
        result_data.append((labels[i], round(scores[i] * 100, 2)))
    
    return render_template("sentiment_result.html", input_sentiment=input_sentiment, result=result_data)

@app.route("/emotion", methods=["GET", "POST"])
def emotion_page():
    return render_template("emotion.html")

@app.route("/emotion_result", methods=["GET", "POST"])
def emotion_result_page():
    input_emotion = request.form['input_emotion']
    emotion_labels = emotion_classifier(input_emotion)
    result_data = [(label['label'].capitalize(), round(label['score'] * 100, 2)) for label in emotion_labels]
   
    return render_template("emotion_result.html", input_emotion=input_emotion, result=result_data)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
