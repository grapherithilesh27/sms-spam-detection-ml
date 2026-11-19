"""SMS spam detection with TF-IDF features and Multinomial Naive Bayes.

Usage:
    python src/spam_detector.py                 # download data, train, evaluate, save model
    python src/spam_detector.py --predict "WIN a FREE prize now!!"
"""
from __future__ import annotations

import argparse
import io
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_PATH = ROOT / "models" / "spam_model.pkl"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Fetch the UCI SMS Spam Collection if it is not already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    path.write_bytes(archive.read("SMSSpamCollection"))
    return path


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df.dropna()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumber ", text)
    text = re.sub(r"[£$€]\s?\d+", " money ", text)
    return re.sub(r"[^a-z\s]", " ", text)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                                      ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("nb", MultinomialNB(alpha=0.1)),
        ]
    )


def train(df: pd.DataFrame, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=seed, stratify=df["label"]
    )
    model = build_pipeline().fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, target_names=["ham", "spam"]),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }
    return model, metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--predict", nargs="+", help="classify one or more messages with the saved model")
    args = parser.parse_args()

    if args.predict:
        model = load_model() if MODEL_PATH.exists() else train(load_dataset(download_dataset()))[0]
        for msg, p, prob in zip(args.predict, model.predict(args.predict), model.predict_proba(args.predict)[:, 1]):
            print(f"[{'SPAM' if p else 'HAM '}] ({prob:.2%} spam)  {msg}")
        return

    df = load_dataset(download_dataset())
    print(f"Loaded {len(df)} messages ({df['label'].mean():.1%} spam)")
    model, m = train(df)
    print(f"\nAccuracy: {m['accuracy']:.4f}\n")
    print(m["report"])
    print("Confusion matrix [[TN FP] [FN TP]]:\n", m["confusion_matrix"])
    save_model(model)
    print(f"\nModel saved to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
