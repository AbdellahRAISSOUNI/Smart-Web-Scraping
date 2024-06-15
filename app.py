from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline, AutoTokenizer
import requests
import re
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'
db = SQLAlchemy(app)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(11), nullable=False)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.Float, nullable=False)

    def __init__(self, video_id, text, sentiment):
        self.video_id = video_id
        self.text = text
        self.sentiment = sentiment

with app.app_context():
    db.create_all()

API_KEY = 'AIzaSyDCx--NQ-T_WE7RDX_c2VsrKLMGzuGQsMg'
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentiment_analyzer = pipeline('sentiment-analysis', model=MODEL_NAME, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form.get('video_url')
    if not video_url:
        return "Please provide a video URL in the POST request.", 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube video URL.", 400

    try:
        comments = fetch_comments(video_id)
        truncated_comments = truncate_comments(comments)
        print(f"Truncated comments: {truncated_comments}")
        sentiment_scores = analyze_sentiment(truncated_comments)
        store_comments(video_id, truncated_comments, sentiment_scores)  # Store in database
        comment_sentiment_pairs = list(zip(truncated_comments, sentiment_scores))
        sentiment_plot = plot_sentiments(sentiment_scores)
        return render_template('results.html', comment_sentiment_pairs=comment_sentiment_pairs, sentiment_plot=sentiment_plot)
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
    truncated_comments = []
    for comment in comments:
        tokens = tokenizer.tokenize(comment)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        truncated_comments.append(tokenizer.convert_tokens_to_string(tokens))
    return truncated_comments

def analyze_sentiment(comments):
    sentiment_scores = []
    for comment in comments:
        tokens = tokenizer.encode(comment, truncation=True, max_length=512)
        if len(tokens) > 512:
            raise ValueError(f"Tokenized comment exceeds max length: {len(tokens)} tokens")
        result = sentiment_analyzer(comment)
        sentiment_scores.append(result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score'])
    return sentiment_scores

def store_comments(video_id, comments, sentiment_scores):
    for comment, sentiment in zip(comments, sentiment_scores):
        new_comment = Comment(video_id=video_id, text=comment, sentiment=sentiment)
        db.session.add(new_comment)
    db.session.commit()

def plot_sentiments(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.hist(sentiment_scores, bins=30, alpha=0.7, color='blue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

if __name__ == '__main__':
    app.run(debug=True)
