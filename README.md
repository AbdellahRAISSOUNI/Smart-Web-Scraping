# YouTube Comment Sentiment Analyzer

This project is a web application that analyzes the sentiment of comments on YouTube videos. It uses Flask for the web framework, SQLAlchemy for database management, and the Hugging Face Transformers library for sentiment analysis.

## Features

- Fetches comments from YouTube videos.
- Analyzes the sentiment of the comments.
- Stores comments and their sentiment scores in a SQLite database.
- Displays graphical sentiment distribution.
- Displays comments along with their sentiment scores and overall sentiment.

## Prerequisites

- Python 3.7+
- Flask
- Flask-SQLAlchemy
- Transformers library by Hugging Face
- Requests
- Matplotlib

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/youtube-comment-sentiment-analyzer.git
    cd youtube-comment-sentiment-analyzer
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1. **Set up your YouTube Data API key:**

    - Obtain an API key from the [Google Cloud Console](https://console.cloud.google.com/).
    - Replace the placeholder API key in `app.py`:

    ```python
    API_KEY = 'YOUR_YOUTUBE_DATA_API_KEY'
    ```

## Usage

1. **Run the application:**

    ```sh
    python app.py
    ```

2. **Open your web browser and navigate to:**

    ```
    http://127.0.0.1:5000/
    ```

3. **Analyze YouTube comments:**

    - Enter the URL of the YouTube video you want to analyze.
    - Select the number of comments to analyze (50, 100, 500, or all comments).
    - Click the "Analyze" button.

4. **View the results:**

    - The application will display the overall sentiment score for the video.
    - It will also show each comment with its corresponding sentiment score.
    - A histogram showing the sentiment distribution will be displayed.

## File Structure

- `app.py`: The main Flask application file.
- `templates/`: Contains HTML templates for the web pages.
  - `index.html`: The main page for inputting the YouTube video URL.
  - `results.html`: The results page displaying the sentiment analysis.
- `requirements.txt`: A list of required Python packages.

## Screenshots (coming soon)

### Home Page

![Home Page](screenshots/home_page.png)

### Results Page

![Results Page](screenshots/results_page.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

If you have any questions or feedback, feel free to contact me at abdellahraissouni@gmail.com.
