from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import whisper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import openai
import matplotlib.pyplot as plt
from transformers import pipeline
import whisper
import openai
import numpy as np
import librosa
from googletrans import Translator
from pyannote.audio import Pipeline
from keybert import KeyBERT
import json
import time
from wordcloud import WordCloud
import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from transformers import pipeline
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob



app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav'}

# Load models
asr_model = whisper.load_model("base")
sentiment_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
translator = Translator()
stop_words = set(stopwords.words('english'))
kw_model = KeyBERT()


# Set your OpenAI API key
openai.api_key = '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'
# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is part of the request 
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Transcription
        transcription = transcribe_audio_video(file_path)
        
        # Multi-language translation
        translated_text = translate_text(transcription, dest_language='es')  # Example: Spanish

        # Topic Extraction
        topics = extract_topics(transcription)

        # Keyword extraction
        keywords = extract_keywords(transcription)

        # Sentiment Analysis
        sentiment = analyze_sentiment(transcription)

        # Emotion Detection
        emotions = detect_emotion(transcription)

        # Calculate audio duration
        duration = calculate_audio_duration(file_path)

        # Detect language
        language = detect_language(transcription)

        # Count words and word frequency
        word_count, word_freq = count_words_and_frequency(transcription)


        # Insight Generation using GPT-3
        insights = generate_insights(transcription)

        # After transcription, add this line to generate the word cloud
        
        generate_wordcloud(transcription)


        create_sentiment_chart(sentiment)

        # Create emotion trend chart
        create_emotion_trend_chart(emotions)

        return render_template('results.html', 
                               transcription=transcription,
                               translated_text=translated_text, 
                               topics=topics,
                               keywords=keywords, 
                               sentiment=sentiment, 
                               emotions=emotions, 
                               insights=insights,
                               duration=duration,
                               language=language,
                               word_count=word_count,
                               word_freq=word_freq,
                               filename=filename)
    return redirect(url_for('index'))
# Transcribe audio/video file using Whisper
def transcribe_audio_video(file_path):
    result = asr_model.transcribe(file_path)
    return result["text"]

def calculate_audio_duration(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    return f"{int(duration)}s"

# Create a mapping of language codes to full language names
LANGUAGE_CODES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'hi': 'Hindi',
    'ar': 'Arabic',
    # Add more languages as needed
}

def detect_language(text):
    translator = Translator()
    detected_lang = translator.detect(text)
    
    # Get the full name of the detected language using the language code
    full_language_name = LANGUAGE_CODES.get(detected_lang.lang, detected_lang.lang)  # Fallback to code if not found
    return full_language_name

from collections import Counter

def count_words_and_frequency(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum()]
    word_count = len(filtered_tokens)
    word_freq = Counter(filtered_tokens)
    return word_count, word_freq.most_common(5)

# Extract key topics from text using LDA
def extract_topics(text):
    # Tokenization and removing stopwords
    tokenized_text = [word_tokenize(text.lower())]
    filtered_texts = [[word for word in tokens if word not in stop_words and word.isalnum()] for tokens in tokenized_text]

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(filtered_texts)
    corpus = [dictionary.doc2bow(tokens) for tokens in filtered_texts]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

    # Extract topics
    raw_topics = lda_model.print_topics(num_words=4)

    # Clean up the topics to show only the words, not the weights
    cleaned_topics = []
    for topic in raw_topics:
        topic_terms = [term.split("*")[1].replace('"', '').strip() for term in topic[1].split(" + ")]
        cleaned_topics.append(", ".join(topic_terms))  # Joining terms into a single string for readability

    return cleaned_topics

# Analyze sentiment from text using VADER
def analyze_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)

# Detect emotions in the text
def detect_emotion(text):
    return emotion_model(text)

# Generate insights from transcription using GPT-3
def generate_insights(text):
    try:
        # 1. Summarize text using spaCy
        summary = summarize_text(text)

        # 2. Sentiment analysis using VADER
        sentiment = analyze_sentiment(text)  # No unpacking needed here

        # 3. Topic extraction using Gensim LDA
        topics = extract_topics(text)

        # Prepare insights dictionary
        insights = (summary)  # Cleaned human-readable topics
                    
        
        return insights
    
    except Exception as e:
        return f"Error generating insights: {e}"


# Summarize the text using SpaCy
def summarize_text(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    # Just a simple summary taking the first 2 sentences
    summary = " ".join([sent.text for sent in sentences[:2]])
    return summary
    
def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

@app.route('/upload', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    language = data.get('language')
    
    if text and language:
        translated_text = translate_text(text, language)
        return jsonify({'translated_text': translated_text})
    else:
        return jsonify({'error': 'Invalid input'}), 400
    
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return keywords


def create_emotion_trend_chart(emotions):
    emotion_scores = [emo['score'] for emo in emotions]
    emotion_labels = [emo['label'] for emo in emotions]

    # Define color gradient based on scores
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(emotion_scores)))

    fig, ax = plt.subplots(figsize=(5, 5))

    # Horizontal bar chart with color gradient
    bars = ax.barh(emotion_labels, emotion_scores, color=colors)

    # Add a grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')

    # Add chart title and labels
    ax.set_title('Emotion Trend Chart', fontsize=16, weight='bold', color='#b32810')
    ax.set_xlabel('Scores', fontsize=12)
    
    # Add data labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', ha='left', fontsize=10, color='#333')

    # Save the figure
    plt.tight_layout()
    plt.savefig('static/images/emotion_trend.png')

def generate_wordcloud(text):
    wordcloud = WordCloud(width=500, height=400, background_color='white').generate(text)
    wordcloud.to_file('static/images/wordcloud.png')  # Save word cloud image


def create_sentiment_chart(sentiment):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiment['pos'], sentiment['neg'], sentiment['neu']]
    colors = ['#66b3ff', '#ff6666', '#99ff99']
    
    # Exploding the slices to create a 3D-like effect
    explode = (0.1, 0.1, 0.1)  # slightly separate all slices

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Adding shadow to make the pie look more 3D
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, shadow=True)
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.savefig('static/images/sentiment_pie_chart.png')



if __name__ == '__main__':
    app.run(debug=True)
