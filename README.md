YouTube Comment Sentiment Analyzer
This is a Flask web application that analyzes the sentiment of comments on YouTube videos. The application fetches comments from a given YouTube video URL, performs sentiment analysis on the comments using the Hugging Face Transformers library, and displays the sentiment distribution along with the individual comment sentiments.
Features

Enter a YouTube video URL to fetch comments
Choose the number of comments to analyze (50, 100, 500, or all comments)
Perform sentiment analysis on the comments using a pre-trained model
Display the sentiment distribution as a histogram
Show the individual comments and their corresponding sentiment scores
Store the analyzed comments and sentiments in a SQLite database

Installation

Clone the repository:

Copygit clone https://github.com/yourusername/youtube-comment-sentiment-analyzer.git

Navigate to the project directory:

Copycd youtube-comment-sentiment-analyzer

Create a virtual environment and activate it:

Copypython -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

Install the required dependencies:

Copypip install -r requirements.txt

Set the YouTube API key in app.py:

pythonCopyAPI_KEY = 'YOUR_YOUTUBE_API_KEY'
Replace 'YOUR_YOUTUBE_API_KEY' with your actual YouTube Data API key. You can obtain an API key from the Google Cloud Console.
Usage

Run the Flask application:

Copypython app.py

Open your web browser and visit http://localhost:5000.
Enter a YouTube video URL in the input field and select the number of comments to analyze.
Click the "Analyze" button to fetch and analyze the comments.
The application will display the sentiment distribution plot, overall sentiment score, and a list of individual comments with their sentiment scores.

Project Structure

app.py: The main Flask application file that handles the routes and sentiment analysis logic.
templates/index.html: The HTML template for the home page where the user enters the YouTube video URL.
templates/results.html: The HTML template for displaying the sentiment analysis results.
static/styles.css: The CSS file for styling the web pages.
comments.db: The SQLite database file for storing analyzed comments and sentiments.

Dependencies
The project uses the following libraries and tools:

Flask
Flask-SQLAlchemy
Transformers (Hugging Face)
Requests
Matplotlib
SQLite

Contributing
Contributions are welcome! If you find any issues or want to add new features, please open an issue or submit a pull request.
License
This project is licensed under the MIT License.
