import re, string, unicodedata
from ftfy import fix_text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
import joblib

# from tensorflow.keras.models import load_model
import pandas as pd


# تنظيف النص
def clean_text(text):
    text = fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def clean_text2(text):
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def get_daily_articles_from_csv(csv_path="news_live.csv"):
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    df.columns = df.columns.str.strip()  # إزالة الفراغات
    df.columns = df.columns.str.replace("\ufeff", "")  # إزالة BOM إذا فيه

    articles = []

    for _, row in df.iterrows():
        articles.append(
            {
                "title": clean_text2(row.get("Title", "No Title")),
                "content": clean_text2(row.get("Content", "No Content")),
                "url": row.get("URL", "#"),
                "source": row.get("Source", "default"),
                "published": row.get("PublishedDate", "Unknown"),
                "image": row.get("ImageURL", None),  # يمكنك استخدامه لاحقًا
            }
        )
    return articles
