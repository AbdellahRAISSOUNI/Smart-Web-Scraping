from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import re
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime
import random
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
from textblob import TextBlob
from collections import Counter

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(11), nullable=False)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.Float, nullable=False)
    emotion = db.Column(db.String(20), nullable=True)
    emotion_score = db.Column(db.Float, nullable=True)
    language = db.Column(db.String(10), nullable=True)  # Ensure this line is present
    timestamp = db.Column(db.DateTime, nullable=True)

with app.app_context():
    db.create_all()

API_KEY = 'AIzaSyDCx--NQ-T_WE7RDX_c2VsrKLMGzuGQsMg'  # Replace with your actual YouTube API key
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
EMOTION_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

emotion_analyzer = pipeline('text-classification', model=EMOTION_MODEL_NAME, return_all_scores=True)

# Initialize BERTopic for topic modeling
vectorizer = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form.get('video_url')
    comment_count = request.form.get('comment_count', 'all')
    if not video_url:
        return "Please provide a video URL in the POST request.", 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube video URL.", 400

    try:
        comments = fetch_comments(video_id)
        if comment_count != 'all':
            comments = comments[:int(comment_count)]
        truncated_comments = truncate_comments(comments)
        
        sentiment_scores, emotions, languages = analyze_comments(truncated_comments)
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        store_comments(video_id, truncated_comments, sentiment_scores, emotions, languages)
        comment_sentiment_pairs = list(zip(truncated_comments, sentiment_scores))
        
        sentiment_plot = plot_sentiments(sentiment_scores)
        wordcloud_plot = generate_wordcloud(truncated_comments)
        positive_wordcloud = generate_wordcloud([c for c, s in comment_sentiment_pairs if s > 0])
        negative_wordcloud = generate_wordcloud([c for c, s in comment_sentiment_pairs if s < 0])
        sentiment_gauge = create_sentiment_gauge(overall_sentiment)
        top_positive, top_negative = get_top_comments(comment_sentiment_pairs)
        
        # Generate mock data for sentiment over time
        comment_dates = [datetime.now().strftime("%Y-%m-%d") for _ in range(len(truncated_comments))]
        
        # Count sentiments
        positive_count = sum(1 for score in sentiment_scores if score > 0)
        negative_count = sum(1 for score in sentiment_scores if score < 0)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        sentiment_counts = [positive_count, neutral_count, negative_count]

        # Count emotions
        emotion_counts = {}
        for emotion, _ in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        emotion_plot = plot_emotions(emotion_counts)

        # Topic modeling
        topics, _ = topic_model.fit_transform(truncated_comments)
        topic_info = topic_model.get_topic_info()

        # Aspect-based sentiment analysis (simplified)
        aspects = ['content', 'editing', 'audio', 'visuals']
        aspect_sentiments = perform_aspect_based_sentiment(truncated_comments, aspects)

        # Language distribution
        language_distribution = {}
        for lang in languages:
            language_distribution[lang] = language_distribution.get(lang, 0) + 1

        # Keyword extraction
        keywords = extract_keywords(truncated_comments)

        return render_template('results.html', 
                               comment_sentiment_pairs=comment_sentiment_pairs, 
                               sentiment_plot=sentiment_plot,
                               wordcloud_plot=wordcloud_plot,
                               positive_wordcloud=positive_wordcloud,
                               negative_wordcloud=negative_wordcloud,
                               sentiment_gauge=sentiment_gauge,
                               overall_sentiment=overall_sentiment,
                               top_positive=top_positive,
                               top_negative=top_negative,
                               comment_dates=comment_dates,
                               sentiment_scores=sentiment_scores,
                               sentiment_counts=sentiment_counts,
                               positive_count=positive_count,
                               negative_count=negative_count,
                               neutral_count=neutral_count,
                               video_id=video_id,
                               youtube_api_key=API_KEY,
                               emotion_plot=emotion_plot,
                               emotions=emotions,
                               topics=topic_info.to_dict(),  # Convert Series to dict for JSON serialization
                               aspect_sentiments=aspect_sentiments,
                               language_distribution=language_distribution,
                               keywords=keywords)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

def extract_video_id(video_url):
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', video_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

def fetch_comments(video_id):
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch comments from YouTube API")
    
    data = response.json()
    for item in data['items']:
        comment_text = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment_text)
    
    return comments

def truncate_comments(comments, max_length=512):
    return [comment[:max_length] for comment in comments]

def analyze_comments(comments):
    sentiment_scores = []
    emotions = []
    languages = []
    for comment in comments:
        sentiment_result = sentiment_analyzer(comment)
        sentiment_scores.append(sentiment_result[0]['score'] if sentiment_result[0]['label'] == 'POSITIVE' else -sentiment_result[0]['score'])
        
        emotion_result = emotion_analyzer(comment)[0]
        top_emotion = max(emotion_result, key=lambda x: x['score'])
        emotions.append((top_emotion['label'], top_emotion['score']))
        
        try:
            lang = detect(comment)
        except:
            lang = 'unknown'
        languages.append(lang)
    
    return sentiment_scores, emotions, languages

def store_comments(video_id, comments, sentiment_scores, emotions, languages):
    for comment, sentiment, (emotion, emotion_score), language in zip(comments, sentiment_scores, emotions, languages):
        new_comment = Comment(video_id=video_id, text=comment, sentiment=sentiment, 
                              emotion=emotion, emotion_score=emotion_score, 
                              language=language, timestamp=datetime.now())
        db.session.add(new_comment)
    db.session.commit()

def plot_sentiments(sentiment_scores):
    fig = go.Figure(data=[go.Histogram(x=sentiment_scores, nbinsx=30)])
    fig.update_layout(title_text='Sentiment Distribution', xaxis_title_text='Sentiment Score', yaxis_title_text='Frequency')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def plot_emotions(emotion_counts):
    fig = go.Figure(data=[go.Bar(x=list(emotion_counts.keys()), y=list(emotion_counts.values()))])
    fig.update_layout(title_text='Emotion Distribution', xaxis_title_text='Emotion', yaxis_title_text='Count')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_wordcloud(comments):
    stop_words = set(stopwords.words('english'))
    comment_words = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(comment_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def create_sentiment_gauge(overall_sentiment):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = overall_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Sentiment"},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "yellow"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': overall_sentiment
            }
        }
    ))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_top_comments(comment_sentiment_pairs, top_n=5):
    sorted_pairs = sorted(comment_sentiment_pairs, key=lambda x: x[1], reverse=True)
    top_positive = sorted_pairs[:top_n]
    top_negative = sorted_pairs[-top_n:][::-1]
    return top_positive, top_negative

def perform_aspect_based_sentiment(comments, aspects):
    aspect_sentiments = {aspect: [] for aspect in aspects}
    for comment in comments:
        for aspect in aspects:
            if aspect in comment.lower():
                sentiment = TextBlob(comment).sentiment.polarity
                aspect_sentiments[aspect].append(sentiment)
    return aspect_sentiments

def extract_keywords(comments):
    stop_words = set(stopwords.words('english'))
    words = ' '.join(comments).lower().split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(words).most_common(20)

if __name__ == '__main__':
    app.run(debug=True)