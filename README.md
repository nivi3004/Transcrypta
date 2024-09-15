**Transcrypta - Audio Transcription and Insights Generation App**

**Project Description:**
This application processes audio files to generate text transcriptions, extract topics and keywords, perform sentiment and emotion analysis, and detect the number of speakers. Additionally, it provides metadata such as the duration of the conversation, language spoken, word count, and word frequency. The app also generates insights, word clouds, and visualizes sentiment/emotion trends.

**Features:**
**Audio Transcription:** Convert audio files to text using Whisper.
**Sentiment Analysis:** Analyze text polarity (positive, negative, neutral) using VADER and TextBlob.
**Emotion Detection:** Detect emotions in the text (anger, joy, sadness, etc.) using the j-hartmann/emotion-english-distilroberta-base model.
**Topic and Keyword Extraction:** Identify key topics using LDA and extract relevant keywords using KeyBERT.
**Metadata Extraction:**
      -Duration of the conversation
      -Language of the conversation
      -Word count and frequency of words
**Word Cloud Generation:** Visualize the most frequent words in the transcription as a word cloud.
**Visualizations:** Generate visual charts for sentiment analysis and emotion trends.

**Setup Instructions**
1. Clone the Repository
git clone <repository-url>
2. Install Dependencies
Navigate to the project folder and install the required packages using:
pip install -r requirements.txt
3. Download Required Models
Whisper: The Whisper model is downloaded automatically when running the code, but ensure your environment supports model loading.
Pyannote speaker diarization model: Follow the instructions on their official page for setup.
4. Run the Application
Run the Flask application:
python app.py
Access the app by navigating to http://127.0.0.1:5000 in your browser.

**Usage:**
**Upload Audio File:**
Navigate to the upload page and select an audio file (.mp3, .mp4, .wav).
**Transcription and Analysis:**
After uploading, the app automatically transcribes the audio, detects the number of speakers, performs sentiment analysis, emotion detection, topic extraction, and generates metadata.
**View Results:**
**The results page will display:**
- Transcribed text
- Sentiment analysis results
- Detected emotions
- Extracted topics and keywords
- Metadata (duration, language, word count, frequency)
- Insights from the transcription
- Word cloud and sentiment/emotion trend charts.

**Project Dependencies:**
**The application relies on various libraries for transcription, NLP tasks, and visualizations. Here's a list of key dependencies:**
Flask: Web framework for running the app.
Whisper: Model for audio-to-text transcription.
VADER Sentiment Analysis: Lightweight sentiment analysis tool.
TextBlob: Polarity analysis for sentiment.
Hugging Face Transformers: Emotion detection using the j-hartmann/emotion-english-distilroberta-base model.
Gensim LDA: Topic extraction using Latent Dirichlet Allocation.
KeyBERT: Keyword extraction from transcriptions.
Google Translator API: Detects the language of the transcription.
Librosa: Audio processing and extracting metadata such as duration.
Matplotlib: Visualization of charts (sentiment and emotion trends).
WordCloud: Generates a word cloud image from the transcription.
Required Models
Whisper Model (base version): For automatic speech recognition.
Emotion Detection Model (j-hartmann/emotion-english-distilroberta-base): Pre-trained RoBERTa model from Hugging Face for emotion classification.

**File Structure:**
/uploads/           # Folder where uploaded audio files are stored
/static/images/     # Folder where generated visualizations are saved (e.g., word cloud, sentiment charts)
/templates/         # HTML templates for Flask app (upload form, results page)
app.py              # Main Flask application
requirements.txt    # Required dependencies for the project
README.md           # Project documentation

**API Usage:**
1. Transcription
The transcription functionality uses Whisper to convert audio to text:

2. Sentiment Analysis
Sentiment is analyzed using VADER, which returns polarity scores for the transcription:

3. Emotion Detection
Emotion detection is performed using a pre-trained Hugging Face model:

4. Topic and Keyword Extraction
Topics and keywords are extracted using Gensim LDA and KeyBERT, respectively:

**Usage Example:**
Upload an audio file: The app will automatically perform transcription and analysis.
Access the transcription: View the transcription along with detected sentiment, emotions, and insights.
Check visualizations: Review the generated word cloud, sentiment, and emotion trend charts.

Contact:
For any questions or feedback, feel free to contact [Nivetha] at [nivethadass2004@gmail.com].
