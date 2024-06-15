# YouTube Comment Sentiment Analyzer

This is a Flask web application that analyzes the sentiment of comments on YouTube videos. The application fetches comments from a given YouTube video URL, performs sentiment analysis on the comments using the Hugging Face Transformers library, and displays the sentiment distribution along with the individual comment sentiments.

## Features

- Enter a YouTube video URL to fetch comments
- Choose the number of comments to analyze (50, 100, 500, or all comments)
- Perform sentiment analysis on the comments using a pre-trained model
- Display the sentiment distribution as a histogram
- Show the individual comments and their corresponding sentiment scores
- Store the analyzed comments and sentiments in a SQLite database

## Installation

1. Clone the repository:
   `git clone https://github.com/yourusername/youtube-comment-sentiment-analyzer.git`

2. Navigate to the project directory:
  `cd youtube-comment-sentiment-analyzer`

3. Create a virtual environment and activate it:
 ` python -m venv env
source env/bin/activate  # On Windows, use env\Scripts\activate`

4. Install the required dependencies:
  `pip install -r requirements.txt`

5. Set the YouTube API key in `app.py`:

```python
API_KEY = 'YOUR_YOUTUBE_API_KEY'
```

Replace 'YOUR_YOUTUBE_API_KEY' with your actual YouTube Data API key. You can obtain an API key from the Google Cloud Console.

## Usage

1. Run the Flask application
  `python app.py`

2. Open your web browser and visit `http://localhost:5000`

3. Enter a YouTube video URL in the input field and select the number of comments to analyze.

4. Click the "Analyze" button to fetch and analyze the comments.

5. The application will display the sentiment distribution plot, overall sentiment score, and a list of individual comments with their sentiment scores.


## Contributing

Contributions are welcome! If you find any issues or want to add new features, please open an issue or submit a pull request.

## Licence
This project is licensed under the MIT License.











