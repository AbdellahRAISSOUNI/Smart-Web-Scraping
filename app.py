from flask import Flask, request, render_template
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import re

app = Flask(__name__)

# Replace 'YOUR_API_KEY' with your actual YouTube Data API v3 key
YOUTUBE_API_KEY = 'AIzaSyDCx--NQ-T_WE7RDX_c2VsrKLMGzuGQsMg'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form.get('video_url')
    if not video_url:
        return "Please provide a video URL in the POST request.", 400

    # Extract the video ID from the URL
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube video URL.", 400

    try:
        comments = fetch_comments(video_id)
        sentiment_scores = analyze_sentiment(comments)
        comment_sentiment_pairs = list(zip(comments, sentiment_scores))
        return render_template('results.html', comment_sentiment_pairs=comment_sentiment_pairs)
    except HttpError as e:
        return f"An HTTP error occurred: {e.resp.status}, {e.content}", e.resp.status
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

def extract_video_id(video_url):
    # Regular expression pattern to extract video ID from URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', video_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

def fetch_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

def analyze_sentiment(comments):
    sentiment_scores = []
    for comment in comments:
        blob = TextBlob(comment)
        sentiment_scores.append(blob.sentiment.polarity)
    return sentiment_scores

if __name__ == '__main__':
    app.run(debug=True)