import pandas as pd
import nltk
import pickle
import re
import os
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import textstat

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# NLTK SETUP
# =========================

def setup_nltk():
    """Download required NLTK data with error handling"""
    required_packages = [
        'punkt',
        'stopwords', 
        'vader_lexicon',
        'averaged_perceptron_tagger'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✅ Downloaded {package}")
        except Exception as e:
            print(f"⚠️ Warning: Could not download {package}: {e}")

# Run setup
setup_nltk()

# Initialize NLTK components
try:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
    print("✅ NLTK components initialized successfully")
except Exception as e:
    print(f"❌ Error initializing NLTK: {e}")
    exit(1)

# =========================
# IMPROVED FEATURE EXTRACTION
# =========================
def extract_advanced_features(text):
    """Extract features from text - IMPROVED for short fake reviews"""
    try:
        text = str(text)
        if not text or text.isspace():
            return get_default_features()
            
        words = text.split()
        
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [text] if text else []
        
        try:
            pos = nltk.pos_tag(words) if words else []
        except:
            pos = []

        features = {}

        # Basic features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
        features['word_count'] = len(words)
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        features['sentence_count'] = len(sentences)

        # Sentiment
        try:
            s = sia.polarity_scores(text)
            features['sentiment'] = s['compound']
            features['extreme_sentiment'] = int(abs(s['compound']) > 0.7)
        except:
            features['sentiment'] = 0
            features['extreme_sentiment'] = 0

        # POS features
        features['adjective_count'] = sum(1 for _, t in pos if t and t.startswith('JJ'))
        features['noun_count'] = sum(1 for _, t in pos if t and t.startswith('NN'))

        # Spam indicators
        features['number_count'] = len(re.findall(r'\d+', text))
        features['first_person_count'] = sum(1 for w in words if w.lower() in ['i', 'me', 'my', 'mine'])
        
        # NEW: Short review red flags
        features['very_short'] = int(len(words) <= 5)  # "worst quality" = 2 words
        features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        # Extreme words check
        extreme_words = ['worst', 'best', 'terrible', 'amazing', 'horrible', 'perfect', 
                        'awful', 'fantastic', 'garbage', 'excellent', 'pathetic', 'outstanding']
        features['extreme_words'] = sum(1 for w in words if w.lower() in extreme_words)

        # Readability
        try:
            features['readability'] = textstat.flesch_reading_ease(text)
        except:
            features['readability'] = 0

        return features
        
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return get_default_features()

def get_default_features():
    """Return default feature values"""
    return {
        'exclamation_count': 0,
        'question_count': 0,
        'caps_ratio': 0,
        'word_count': 0,
        'unique_word_ratio': 0,
        'sentence_count': 0,
        'sentiment': 0,
        'extreme_sentiment': 0,
        'adjective_count': 0,
        'noun_count': 0,
        'number_count': 0,
        'first_person_count': 0,
        'very_short': 0,
        'all_caps_words': 0,
        'extreme_words': 0,
        'readability': 0
    }

# =========================
# IMPROVED PREPROCESSING
# =========================
def preprocess_with_features(text):
    """Preprocess text and add feature tokens - IMPROVED"""
    try:
        features = extract_advanced_features(text)

        # Basic text preprocessing
        text_lower = str(text).lower()
        text_clean = re.sub('[^a-zA-Z]', ' ', text_lower)
        
        # Stemming
        try:
            words = [stemmer.stem(w) for w in text_clean.split() if w and w not in stop_words]
        except:
            words = [w for w in text_clean.split() if w]

        # Add feature-based tokens
        tokens = []

        # Existing features
        if features['exclamation_count'] > 2:
            tokens.append("high_exclamation")
        if features['caps_ratio'] > 0.15:
            tokens.append("high_caps")
        if features['extreme_sentiment']:
            tokens.append("extreme_sentiment")
        if features['unique_word_ratio'] < 0.5 and features['unique_word_ratio'] > 0:
            tokens.append("low_diversity")
        if features['adjective_count'] > features['noun_count']:
            tokens.append("adjective_heavy")
        if features['first_person_count'] > 2:
            tokens.append("personal_experience")
        if features['number_count'] > 1:
            tokens.append("specific_details")
        if features['readability'] > 70:
            tokens.append("very_easy_read")
        elif features['readability'] < 30 and features['readability'] > 0:
            tokens.append("hard_to_read")
        
        # NEW: Short fake review detection
        if features['very_short'] and features['extreme_words'] > 0:
            tokens.append("short_extreme")  # "worst quality" pattern
        if features['all_caps_words'] > 0:
            tokens.append("has_caps_words")
        if features['extreme_words'] >= 2:
            tokens.append("multiple_extreme")  # "worst terrible quality"
        
        # Special handling for very short reviews with extreme sentiment
        if len(words) <= 3 and abs(features['sentiment']) > 0.5:
            tokens.append("ultra_short_emotional")

        return " ".join(words + tokens)
        
    except Exception as e:
        print(f"⚠️ Error in preprocessing: {e}")
        return str(text)

# =========================
# LOAD DATASET
# =========================

# Try to load the large dataset first
dataset_files = ["reviews_large_dataset.csv", "reviews_improved.csv", "reviews.csv"]
data = None

for filename in dataset_files:
    if os.path.exists(filename):
        try:
            data = pd.read_csv(filename)
            print(f"✅ Loaded {len(data)} reviews from {filename}")
            break
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")

if data is None:
    print("❌ No dataset found! Please provide reviews.csv")
    exit(1)

# Validate columns
if 'review' not in data.columns or 'label' not in data.columns:
    print("❌ CSV must contain 'review' and 'label' columns")
    exit(1)

# Apply preprocessing
print("🔄 Preprocessing reviews...")
data['clean_review'] = data['review'].apply(preprocess_with_features)

# Show some examples
print("\n📊 Sample preprocessing:")
for i in range(min(3, len(data))):
    print(f"\nOriginal: {data['review'].iloc[i][:80]}")
    print(f"Processed: {data['clean_review'].iloc[i][:80]}")
    print(f"Label: {'FAKE' if data['label'].iloc[i] == 1 else 'GENUINE'}")

# =========================
# TF-IDF
# =========================
print("\n🔄 Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=3000,  # Increased from 2500
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,  # Slightly more permissive
    sublinear_tf=True
)

X = vectorizer.fit_transform(data['clean_review'])
y = data['label']

print(f"📊 Feature matrix shape: {X.shape}")
print(f"📊 Total samples: {len(y)}")
print(f"📊 Fake reviews: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"📊 Genuine reviews: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# =========================
# MODEL TRAINING
# =========================
print("\n🔄 Training Random Forest model...")

# Use better parameters for larger dataset
model = RandomForestClassifier(
    n_estimators=200,  # Increased from 100
    max_depth=15,      # Increased from 10
    min_samples_split=5,
    min_samples_leaf=2,  # Reduced from 3
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Split data if we have enough samples
if len(data) >= 20:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    print("\n📊 TRAINING SET PERFORMANCE:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.3f}")
    
    print("\n📊 TEST SET PERFORMANCE:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"], zero_division=0))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives (Genuine predicted as Genuine): {cm[0][0]}")
    print(f"False Positives (Genuine predicted as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake predicted as Genuine): {cm[1][0]}")
    print(f"True Positives (Fake predicted as Fake): {cm[1][1]}")
else:
    print("⚠️ Dataset too small for train/test split. Using all data.")
    model.fit(X, y)

# =========================
# SAVE MODEL
# =========================
try:
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    print("\n✅ Model & Vectorizer saved successfully")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_review(text):
    """Predict if a review is fake or genuine"""
    try:
        processed = preprocess_with_features(text)
        vec = vectorizer.transform([processed])
        
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        features = extract_advanced_features(text)
        
        return {
            "prediction": "FAKE" if pred == 1 else "GENUINE",
            "confidence": round(max(prob) * 100, 2),
            "fake_probability": round(prob[1] * 100, 2),
            "genuine_probability": round(prob[0] * 100, 2),
            "explanation": {
                "exclamations": features['exclamation_count'],
                "caps_ratio": round(features['caps_ratio'], 2),
                "sentiment": round(features['sentiment'], 3),
                "word_count": features['word_count'],
                "extreme_words": features['extreme_words'],
                "processed_text": processed[:100]
            }
        }
    except Exception as e:
        return {
            "prediction": "ERROR",
            "confidence": 0,
            "fake_probability": 0,
            "genuine_probability": 0,
            "explanation": {"error": str(e)}
        }

# =========================
# TESTING
# =========================
print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

tests = [
    "worst quality",
    "WORST QUALITY!!!",
    "terrible quality",
    "AMAZING!!! BEST PRODUCT EVER!!! MUST BUY!!!",
    "horrible quality",
    "GARBAGE QUALITY!!!",
    "I've used this for 2 weeks. Battery is average but camera is good.",
    "This product is okay. Does what it says but nothing special.",
    "The quality is below average. Had it for two weeks and already seeing issues.",
    "Quality seems questionable. Some parts feel flimsy."
]

for i, t in enumerate(tests, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {t}")
    result = predict_review(t)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Fake probability: {result['fake_probability']}%")
    print(f"Genuine probability: {result['genuine_probability']}%")
    print(f"Features: {result['explanation']}")

print("\n" + "="*60)
print("✅ Training completed successfully!")
print("="*60)
print("\n💡 Next steps:")
print("1. Run: python app.py")
print("2. Visit: http://localhost:5000")
print("3. Test with: 'worst quality' (should show FAKE)")
print("4. Test with: 'The quality is below average' (should show GENUINE)")