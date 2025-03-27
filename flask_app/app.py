import os
from flask import Flask, render_template, request 
import mlflow
import pickle
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Initialize Flask app with explicit template path
TEMPLATE_DIR = os.path.abspath("flask_app/templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Set up MLflow tracking
mlflow.set_tracking_uri('https://dagshub.com/rushikeshnayakavadi/Sentiment-Analysis.mlflow')
dagshub.init(repo_owner='rushikeshnayakavadi', repo_name='Sentiment-Analysis', mlflow=True)

# Create a custom registry for Prometheus
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Load ML model
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name)

    if not versions:
        print(f"No registered versions found for model: {model_name}")
        return None

    return versions[0].version

model_version = get_latest_model_version(model_name)
if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
else:
    model = None

# Load vectorizer
vectorizer_path = os.path.join("models", "vectorizer.pkl")
if os.path.exists(vectorizer_path):
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
else:
    vectorizer = None
    print("Vectorizer not found!")

# Debugging: Print working directory and available templates
print("Current Working Directory:", os.getcwd())
print("Templates Folder Content:", os.listdir(TEMPLATE_DIR))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    # Debug: Check if index.html exists
    template_path = os.path.join(TEMPLATE_DIR, "index.html")
    if not os.path.exists(template_path):
        return "Error: index.html not found!", 500

    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    text = normalize_text(text)

    if vectorizer is None or model is None:
        return "Error: Model or Vectorizer is missing!", 500

    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)
    prediction = "Positive" if result[0] == 1 else "Negative"  # Convert to label

    PREDICTION_COUNT.labels(prediction=prediction).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
