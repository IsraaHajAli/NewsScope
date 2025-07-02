from flask import Flask, request, jsonify, render_template
import random
import numpy as np
from tensorflow.keras.models import load_model
import traceback
import torch
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ftfy import fix_text
import spacy
from transformers import DistilBertForSequenceClassification
import smtplib
from email.mime.text import MIMEText
from flask import request, redirect, flash
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np
import nltk

# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tf_keras_vis.saliency import Saliency
import os
from utils import clean_text
from flask import Flask, request, jsonify, render_template
import subprocess
import fasttext
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer


try:

    subprocess.run(
        [
            "javac",
            "-cp",
            "News_Crawl/json-20230227.jar;News_Crawl/jsoup-1.20.1.jar;News_Crawl",
            "News_Crawl/*.java",
        ],
        check=True,
    )

    subprocess.Popen(
        [
            "java",
            "-cp",
            ".;News_Crawl/json-20230227.jar;News_Crawl/jsoup-1.20.1.jar;News_Crawl",
            "Main",
        ]
    )
    print("✅ Java news fetcher compiled and started in background.")
except Exception as e:
    print("❌ Could not compile or start Java fetcher:", e)


try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

max_len = 150
embedding_dim = 100


tokenizer = DistilBertTokenizerFast.from_pretrained("Ensemble_Files/distilbert_saved")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "Ensemble_Files/distilbert_saved"
)
bert_model.eval()

bilstm_model = load_model("Ensemble_Files/my_bilstm_model.h5")
w2v_model = Word2Vec.load("Ensemble_Files/word2vec_model.bin")


fasttext_model = fasttext.load_model("Ensemble_Files/fasttext_model.bin")
bilstm_fasttext_model = load_model("Ensemble_Files/my_bilstm_model.h5")


nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

explanatory_reasons = {
    "fake": "Suggests the content may be fabricated or misleading.",
    "hoax": "Used to label something as intentionally deceptive.",
    "scam": "Indicates manipulation or fraud.",
    "bullshit": "Slang expressing disbelief or falsehood.",
    "propaganda": "Implies agenda-driven or biased messaging.",
    "agenda": "Suggests hidden motives behind the content.",
    "lie": "Attacks truthfulness of the content.",
    "manipulated": "Indicates altered or deceptive information.",
    "twisted": "Suggests intentional distortion.",
    "damn": "Emotional or informal language, weakens credibility.",
    "wtf": "Slang; emotionally reactive and informal tone.",
    "liar": "Discredits source by attacking honesty.",
    "trash": "Derogatory term implying low quality or falsehood.",
    "sheeple": "Used to describe blindly obedient people; emotional.",
    "you won’t believe": "Clickbait language, often in fake articles.",
    "wake up": "Populist language urging disbelief of mainstream views.",
    "must watch": "Classic clickbait phrase designed to provoke curiosity or urgency.",
    "breaking news": "Often abused to exaggerate or fabricate urgency.",
    "watch video": "Used to lure users into consuming emotionally charged content.",
    "click here": "Clickbait CTA, frequently found in misleading or spammy articles.",
    "truth revealed": "Implies hidden information just uncovered; common in fake content.",
    "deep state": "Conspiracy theory term; undermines trust in institutions.",
    "mainstream media": "Used in a dismissive tone to discredit reliable sources.",
    "exposed video": "Sensational phrasing suggesting hidden truths.",
    "hidden truth": "Implies that the truth has been concealed; common in conspiracy narratives.",
    "media won’t show": "Claims suppression of truth; hallmark of fake narratives.",
    "bombshell report": "Over-the-top dramatic term often found in misleading titles.",
    "sheeple wake": "Emotionally manipulative; suggests others are blindly following lies.",
    "global elites": "Used in conspiracies suggesting a hidden controlling power.",
    "secret documents": "Implying leaked or hidden content to create distrust.",
    "exclusive report": "Overused phrase to fake credibility and uniqueness.",
    "shocking details": "Sensationalist; exaggerates or fabricates emotional impact.",
    "scam alert": "Emotionally charged; creates fear of being deceived.",
    "cover up": "Implies intentional hiding of truth; frequent in conspiracies.",
    "red pill": "Conspiracy term urging people to 'see the truth' behind deception.",
    "alternative media": "Used to elevate unverified sources over trusted ones.",
    "blacklisted news": "Claims suppression or censorship; common in fake posts.",
    "no one talks": "Suggests secret truth is being ignored; emotionally persuasive.",
    "banned video": "Implies censorship to make content appear forbidden and truthful.",
    "viral truth": "False sense of mass validation for unverified content.",
    "unbelievable video": "Clickbait trigger phrase exploiting shock value.",
    "wake up call": "Manipulative phrasing to suggest urgency in realizing deception.",
    "what they don’t want": "Suggests secrecy or censorship; classic fake news tactic.",
    "shocking": "Emotionally charged and designed to provoke overreaction.",
    "exposed": "Used to imply hidden truths being revealed, typical in conspiracies.",
    "alert": "Adds urgency or fear; common in scam-like or misleading posts.",
    "leaked": "Used to give illusion of exclusivity, often without evidence.",
    "proof": "Falsely implies definitive evidence; common in fake claims.",
    "revealed": "Sensational phrasing to suggest something was hidden from the public.",
    "viral": "Used to fake popularity or legitimacy of unverified content.",
    "uncovered": "Implies that someone exposed hidden wrongdoing.",
    "busted": "Colloquial and sensational; implies dramatic exposure.",
    "censored": "Often used to falsely claim content is being suppressed.",
    "they don’t want you to know": "Manipulative phrasing suggesting censorship or secrecy.",
    "wake up america": "Emotionally manipulative; appeals to patriotism and outrage.",
    "massive fraud": "Strong accusation often used without solid evidence.",
    "media blackout": "Claims intentional suppression by mainstream sources.",
    "the truth about": "Common in manipulative articles presenting opinion as fact.",
    "crisis actor": "Conspiracy term used to discredit victims of events.",
    "false flag": "Conspiracy label suggesting staged or fake events.",
    "whistleblower": "Used to add fake credibility to dubious sources.",
    "undercover video": "Used to falsely imply legitimacy and secrecy.",
    "freedom of speech": "Often misused to justify spreading misinformation.",
    "fact check": "Ironically used in fake posts to simulate trustworthiness.",
    "they lied": "Direct attack on official narratives, often baseless.",
}

trusted_reasons = {
    "confirmed": "Shows official or verified information.",
    "officials": "Implies formal authority or governmental source.",
    "report": "Implies info is based on documented data.",
    "reportedly": "Suggests indirect but formal reporting.",
    "study": "Suggests scientific or academic support.",
    "research": "Backed by investigation or experimentation.",
    "published": "Appeared in recognized source or medium.",
    "evidence": "Sign of supporting facts.",
    "source": "Indicates reference to original information.",
    "percent": "Use of statistics gives credibility.",
    "figures": "Numerical data enhances objectivity.",
    "statistics": "Indicates data-based reasoning.",
    "agency": "Implies an organized, possibly governmental body.",
    "government report": "Highly formal and institutional data.",
    "peer-reviewed": "Endorsed by academic community.",
    "experts": "Shows reliance on authoritative opinion.",
    "ministry": "Indicates government-level source.",
    "spokesperson": "Represents official statements.",
    "journal": "Suggests scientific publication.",
    "UN": "Global institutional authority.",
    "WHO": "World Health Organization; trusted medical source.",
    "NATO": "Military and political alliance; trusted geopolitical info.",
    "data shows": "Grounds the statement in measurable facts.",
    "validated": "Confirmed through formal process.",
    "survey": "Data collected systematically.",
    "transcript": "Verbatim record, ensures accuracy.",
    "records": "Historical or factual documentation.",
    "historical data": "Backed by long-term records.",
    "scholars": "Academic authority.",
    "committee": "Formal group decision or investigation.",
    "panel": "Group of experts; formal evaluation.",
    "white house": "Official government reference adds credibility.",
    "washington reuters": "Mention of a known agency increases legitimacy.",
    "north korea": "Geopolitical topic often covered in real news.",
    "prime minister": "Political titles often indicate formal reporting.",
    "told reuters": "Quoting a major news source adds trustworthiness.",
    "supreme court": "Legal and official institution, used in factual articles.",
    "united nations": "Official international reference, boosts credibility.",
    "national security": "Formal terminology often found in real policy discussions.",
    "official said": "Attribution to verified sources; common in real journalism.",
    "state department": "Mentions of government departments add formality.",
    "european union": "Global institution mentioned in formal news.",
    "house representatives": "U.S. government entity, adds authenticity.",
    "presidential election": "Neutral political event term, common in real news.",
    "attorney general": "Legal authority reference, increases trust.",
    "foreign minister": "Official title in diplomatic context.",
    "human rights": "Often discussed in verified international news.",
    "south korea": "Real geopolitical reference typical of fact-based reporting.",
    "white house": "Official government reference adds credibility.",
    "washington reuters": "Mention of a known agency increases legitimacy.",
    "north korea": "Geopolitical topic often covered in real news.",
    "prime minister": "Political titles often indicate formal reporting.",
    "told reuters": "Quoting a major news source adds trustworthiness.",
    "supreme court": "Legal and official institution, used in factual articles.",
    "united nations": "Official international reference, boosts credibility.",
    "national security": "Formal terminology often found in real policy discussions.",
    "official said": "Attribution to verified sources; common in real journalism.",
    "state department": "Mentions of government departments add formality.",
    "european union": "Global institution mentioned in formal news.",
    "house representatives": "U.S. government entity, adds authenticity.",
    "presidential election": "Neutral political event term, common in real news.",
    "attorney general": "Legal authority reference, increases trust.",
    "foreign minister": "Official title in diplomatic context.",
    "human rights": "Often discussed in verified international news.",
    "south korea": "Real geopolitical reference typical of fact-based reporting.",
    "according to": "Indicates attribution to a source; common in real journalism.",
    "press release": "Official communication from a verified entity.",
    "nonprofit organization": "Often tied to factual advocacy or studies.",
    "verified sources": "Implies information has been checked.",
    "official records": "Refers to documented history or evidence.",
    "department of justice": "Formal U.S. government body, boosts trust.",
    "legislation": "Legal terminology, typical of real news.",
    "federal agency": "Implies formal U.S. institutional source.",
    "intelligence report": "Usually implies state-level findings.",
    "scientific journal": "Trusted peer-reviewed outlet.",
    "court documents": "Legal evidence, common in verified articles.",
    "supreme court ruling": "Cites a legal event; high reliability.",
    "non-governmental organization": "Common in real reporting on humanitarian issues.",
    "independent watchdog": "Implies oversight or auditing authority.",
    "ombudsman": "A formal representative of public interest.",
    "justice department": "Imparts institutional credibility.",
    "news conference": "Official setting for formal statements.",
    "environmental protection agency": "Trusted U.S. federal body on science/policy.",
    "census bureau": "Source of trusted demographic data.",
    "academic institution": "Implies source is educational and research-based.",
    "freedom of information act": "Suggests data obtained legally from the state.",
    "university study": "Conveys scientific or academic research.",
    "official announcement": "Imparts a formal communication tone.",
    "public health agency": "Adds scientific authority to health news.",
    "civil rights organization": "Implies social justice credibility.",
    "audited report": "Confirms reviewed financial/factual data.",
    "nonpartisan group": "Suggests neutrality and credibility.",
    "policy paper": "Common in official or academic settings.",
    "economic indicators": "Numerical backing typical in real economic reporting.",
    "factual basis": "Highlights grounding in verifiable evidence.",
}


app = Flask(__name__)
app.secret_key = "1492002key"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


from utils import get_daily_articles_from_csv


def normalize_source_name(source):
    return source.lower().replace(" ", "_").replace("-", "_")


from concurrent.futures import ThreadPoolExecutor


@app.route("/analyze", methods=["POST"])
@app.route("/analyze/", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    cleaned = clean_text(text)

    print("🔍 RAW TEXT:", text[:100])
    print("🧼 CLEANED TEXT:", cleaned[:100])
    print("✂️ Length after cleaning:", len(cleaned))

    prob = predict_distilbert(cleaned)
    label = "Real" if prob > 0.5 else "Fake"
    # confidence = round(prob * 100 if label == "Real" else (1 - prob) * 100, 2)

    if not cleaned or len(cleaned.strip()) < 10:
        return "⚠️ No valid content to analyze.", 200

    explanation = explain_bilstm_with_sentences_and_phrases(cleaned)
    explanation_str = str(explanation) if explanation else "Explanation missing."

    return jsonify(
        {
            "label": label,
            "Decision Explanation": explanation_str,
        }
    )

    # exp = explain_distilbert_with_phrases(cleaned, truncate=True)

    # return jsonify(
    #     {
    #         "label": label,
    #         "reason": exp,
    #     }
    # )


from random import shuffle


@app.route("/daily-news")
def daily_news():
    try:
        articles = get_daily_articles_from_csv("news_live.csv")

        shuffle(articles)

        final_articles = []

        for article in articles:
            print("keeeeeeeeeeeeeeeeeeyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
            print(article.keys())

            content = article.get("content")

            if len(content) > 2000:
                content = content[:2000]

            final_articles.append(
                {
                    "title": article.get("title", "No Title"),
                    "content": content,
                    "url": article.get("url", "#"),
                    "source": article.get("source", "default"),
                    "published": article.get("published", "Unknown"),
                    "image": article.get("image", None),  # إذا بدك تضيفي الصورة لاحقًا
                }
            )

        return render_template("daily_news.html", articles=final_articles)

    except Exception as e:
        return f"Error: {str(e)}"


def normalize_phrase(text):
    return re.sub(r"[^\w\s]", "", text.lower())


# @app.route("/daily-news")
# def daily_news():
#     try:
#         articles = get_daily_articles_from_csv("News_Crawl/news_live.csv")
#         results = []

#         def process_article(article):
#             text = article["content"]
#             cleaned = clean_text(text)

#             avg_prob = predict_distilbert(cleaned)
#             # prob_bilstm = predict_bilstm(cleaned)
#             # avg_prob = 0.6 * prob_bert + 0.4 * prob_bilstm

#             num = avg_prob + 0.5

#             if num > 0.98:
#                 num = 0.98

#             label = "Real" if num > 0.5 else "Fake"


#             # confidence = round(
#             #     avg_prob * 100 if label == "Real" else (1 - avg_prob) * 100, 2
#             # )

#             return {
#                 "title": article["title"],
#                 "content": article["content"],
#                 "url": article["url"],
#                 "label": label,
#                 "source": article.get("source", "default"),
#                 "published": article.get("published", "Unknown"),
#                 # "confidence": confidence,
#             }


#         with ThreadPoolExecutor(max_workers=8) as executor:
#             results = list(executor.map(process_article, articles))

#         random.shuffle(results)
#         return render_template("daily_news.html", articles=results)

#     except Exception as e:
#         print("❌ Error in /daily-news:", e)
#         return "Internal Server Error", 500


@app.route("/send-message", methods=["POST"])
def send_message():
    try:
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # إعداد الرسالة
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg = MIMEText(body)
        msg["Subject"] = "New Contact Message from NewsScope"
        msg["From"] = email
        msg["To"] = "israajdali9@gmail.com"

        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = "israajdali9@gmail.com"
        smtp_password = "lqyo dvve puil xfpm"

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        flash("Message sent successfully!", "success")
        return redirect("/contact")

    except Exception as e:
        print("Error sending message:", e)
        flash("Error sending message. Try again later.", "error")
        return redirect("/contact")


###############################################################################################################################################
################################################################ Predictions ###############################################################
###############################################################################################################################################


def predict_distilbert(text):
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()


def predict_bilstm(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    vecs = []
    for word in tokens[:max_len]:
        if word in w2v_model.wv:
            vecs.append(w2v_model.wv[word])
        else:
            vecs.append(np.zeros(embedding_dim))
    while len(vecs) < max_len:
        vecs.append(np.zeros(embedding_dim))
    vecs = np.array(vecs).reshape(1, max_len, embedding_dim)
    prob = bilstm_model.predict(vecs)[0][0]
    return prob


############################################### here add the BiLSTM with FastText ###############################################


def preprocess_text_fasttext(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    return tokens


def vectorize_fasttext(tokens):
    vecs = []
    for word in tokens[:max_len]:
        vecs.append(fasttext_model.get_word_vector(word))
    while len(vecs) < max_len:
        vecs.append(np.zeros(embedding_dim))
    return np.array(vecs).reshape(1, max_len, embedding_dim)


def predict_bilstm_fasttext(text):
    tokens = preprocess_text_fasttext(text)
    vec_input = vectorize_fasttext(tokens)
    prob = bilstm_fasttext_model.predict(vec_input)[0][0]
    return prob


###############################################################################################################################################
###################################################### Ensemble Prediction [Soft Voting] ######################################################
###############################################################################################################################################


def ensemble_predict(text):
    text = clean_text(text)
    prob_bert = predict_distilbert(text)  # 0.9912
    prob_bilstm = predict_bilstm(text)  # 0.9659
    pred_fasttext = predict_bilstm_fasttext(text)  # 0.9469

    # حساب المتوسط المرجح
    # prob_bert = 0.9912
    # prob_bilstm = 0.9717
    # pred_fasttext = 0.9469

    # sum = 0.9912 + 0.9717  + 0.9469 = 2.9098

    avg_prob = (
        (0.9912 / 2.9098) * prob_bert
        + (0.9717 / 2.9098) * prob_bilstm
        + (0.9469 / 2.9098) * pred_fasttext
    )

    print("🧠 DistilBERT Prob:", prob_bert)
    print("🧠 BiLSTM Prob:", prob_bilstm)
    print("📊 Average Prob:", avg_prob)

    prediction = 1 if avg_prob > 0.5 else 0
    print(
        f"DistilBERT: {prob_bert:.3f}, BiLSTM: {prob_bilstm:.3f}, Average: {avg_prob:.3f}"
    )
    return prediction


#################################### here add the hard voting ensemble
#
#
#


def hard_voting_ensemble(text):
    pred_bert = predict_distilbert(text)
    pred_w2v = predict_bilstm(text)
    pred_fasttext = predict_bilstm_fasttext(text)

    votes = [int(pred_bert > 0.5), int(pred_w2v > 0.5), int(pred_fasttext > 0.5)]
    final_prediction = 1 if sum(votes) >= 2 else 0

    print(f"🔢 Votes => BERT: {votes[0]}, W2V: {votes[1]}, FastText: {votes[2]}")
    print(f"🎯 Final Hard Voting Result: {'Real' if final_prediction else 'Fake'}")

    return final_prediction


#
#
#
################################################################################################################################################################
############################################################# XAI Functions [BiLSTM with Word2Vec] #############################################################
################################################################################################################################################################


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words
    ]
    return tokens


def vectorize(tokens):
    vec = []
    for word in tokens:
        if word in w2v_model.wv:
            vec.append(w2v_model.wv[word])
        else:
            vec.append(np.zeros(embedding_dim))
    while len(vec) < max_len:
        vec.append(np.zeros(embedding_dim))
    return np.array(vec[:max_len]).reshape(1, max_len, embedding_dim)


from ftfy import fix_text


def extract_phrases(text):
    text = fix_text(text)
    text = re.sub(r"\s+", " ", text.strip())
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def explain_bilstm_with_sentences_and_phrases(text):
    saliency = Saliency(bilstm_model)

    def score_function(output):
        return output

    results = []
    phrases = extract_phrases(text)
    print(f"Found {len(phrases)} phrases")
    for phrase in phrases[:8]:
        tokens = preprocess_text(phrase)
        if not tokens:
            continue
        phrase_input = vectorize(tokens)
        saliency_map = saliency(score_function, phrase_input)
        score = np.sum(np.abs(saliency_map))
        results.append(("Phrase", phrase.strip(), score))

    results = sorted(results, key=lambda x: x[2], reverse=True)

    filtered_explanatory = []
    for typ, content, score in results:
        if any(k in content.lower() for k in explanatory_reasons):
            filtered_explanatory.append((typ, content, score))

    filtered_trusted = []
    for typ, content, score in results:
        if any(k in content.lower() for k in trusted_reasons):
            filtered_trusted.append((typ, content, score))

    predicted_label = predict_bilstm(text)

    explanation_output = ""

    if predicted_label == 1:
        if filtered_trusted:
            explanation_output += "This article is classified as Real because it contains phrases like:<br>"
            for typ, content, score in filtered_trusted[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"
                for word in trusted_reasons:
                    if word in content.lower():
                        reason = trusted_reasons[word]
                        explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase contains the keyword '<strong>{word}</strong>' → {reason}<br>"
                        break
        else:
            explanation_output += "No strong trusted phrases were found, but overall the article seems Real."
    else:
        if filtered_explanatory:
            explanation_output += (
                "This article is classified as Fake due to phrases like:<br>"
            )
            for typ, content, score in filtered_explanatory[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"
                for word in explanatory_reasons:
                    if word in content.lower():
                        reason = explanatory_reasons[word]
                        explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase contains the keyword '<strong>{word}</strong>' → {reason}<br>"
                        break
        else:
            explanation_output += (
                "No fake indicators found, but the article still seems suspicious."
            )

    return explanation_output


################################################################################################################################################################
############################################################# XAI Functions [DistilBERT] #############################################################
################################################################################################################################################################


def explain_distilbert_with_phrases(text, truncate=False, max_length=512):

    cleaned = clean_text(text)

    # if truncate:
    #     inputs = tokenizer(cleaned, truncation=True, max_length=512, return_tensors="pt")
    #     explainer = SequenceClassificationExplainer(model=bert_model, tokenizer=tokenizer)
    #     attribution = explainer.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # fallback only
    # else:
    #     explainer = SequenceClassificationExplainer(model=bert_model, tokenizer=tokenizer)
    #     attribution = explainer(cleaned)

    explainer = SequenceClassificationExplainer(model=bert_model, tokenizer=tokenizer)

    if truncate:
        encoding = tokenizer(cleaned, truncation=True, max_length=max_length)
        truncated_text = tokenizer.decode(
            encoding["input_ids"], skip_special_tokens=True
        )
        attribution = explainer(truncated_text)
        predicted_label = predict_distilbert(truncated_text)
    else:
        attribution = explainer(cleaned)
        predicted_label = predict_distilbert(text)

    results = []

    phrases = extract_phrases(text)
    print(f"Found {len(phrases)} phrases\n")

    for phrase in phrases:
        score = sum([s for w, s in attribution if w in phrase])
        results.append(("Phrase", phrase.strip(), score))

    results = sorted(results, key=lambda x: x[2], reverse=True)

    filtered_explanatory = []
    for typ, content, score in results:
        if any(k in content.lower() for k in explanatory_reasons):
            filtered_explanatory.append((typ, content, score))

    filtered_trusted = []
    for typ, content, score in results:
        if any(k in content.lower() for k in trusted_reasons):
            filtered_trusted.append((typ, content, score))

    predicted_label = predict_distilbert(text)

    print(
        f"🧠 DistilBERT Probability: {predicted_label:.3f} --> {'Real ✅' if predicted_label > 0.5 else 'Fake 🚫'}\n"
    )

    if predicted_label > 0.5:
        print("🔷 Trusted & Formal Evidence Found (Real Articles):\n")
        if filtered_trusted:
            for typ, content, score in filtered_trusted[:10]:
                print(f"📌 Phrase: {content.strip()}")
                print(f"   🔢 Contribution Score: {score:.4f}")
                for word in trusted_reasons:
                    if word in content.lower():
                        reason = trusted_reasons[word]
                        print(
                            f"   🔍 Why it matters: This phrase contains the keyword '{word}' → {reason}"
                        )
                        break
                print("-----------------------------")
        else:
            print("⚠️ No strong formal evidence detected.")
    else:
        print("🔥 Top Interpretable Units (Fake/Misleading Articles):\n")
        if filtered_explanatory:
            for typ, content, score in filtered_explanatory[:10]:
                print(f"📌 Phrase: {content.strip()}")
                print(f"   🔢 Contribution Score: {score:.4f}")
                for word in explanatory_reasons:
                    if word in content.lower():
                        reason = explanatory_reasons[word]
                        print(
                            f"   🔍 Why it matters: This phrase contains the keyword '{word}' → {reason}"
                        )
                        break
                print("-----------------------------")
        else:
            print("⚠️ No clear fake indicators found using known keywords.\n")
            print("💡 Showing top 5 phrases with highest contribution scores:\n")
            for typ, content, score in results[:5]:
                print(f"📌 Phrase: {content.strip()}")
                print(f"   🔢 Contribution Score: {score:.4f}")
                print(
                    "   🔍 Why it matters: This phrase may contain strong language or emotional impact."
                )
                print("-----------------------------")

    explanation_output = ""

    if predicted_label > 0.5:
        if filtered_trusted:
            explanation_output += "This article is classified as Real because it contains phrases like:<br>"
            for typ, content, score in filtered_trusted[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"

                for word in trusted_reasons:
                    if word in content.lower():
                        reason = trusted_reasons[word]
                        explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase contains the keyword '<strong>{word}</strong>' → {reason}<br>"
                        break
        else:
            # explanation_output += "This article is classified as Real, but no predefined trusted keywords were detected.<br>"
            # explanation_output += "However, here are the most influential phrases detected by the model:<br>"
            for typ, content, score in results[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"
                explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase may indicate factual, neutral, or formal language based on the model's interpretation.<br>"

    else:
        if filtered_explanatory:
            explanation_output += (
                "This article is classified as Fake due to phrases like:<br>"
            )
            for typ, content, score in filtered_explanatory[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"

                for word in explanatory_reasons:
                    if word in content.lower():
                        reason = explanatory_reasons[word]
                        explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase contains the keyword '<strong>{word}</strong>' → {reason}<br>"
                        break
        else:
            explanation_output += "This article is classified as Fake, and here are the most suspicious phrases based on the model's saliency:<br>"
            for typ, content, score in results[:3]:
                explanation_output += f"&nbsp;&nbsp;&bull; <strong>{content.strip()}</strong> (score: {score:.3f})<br>"
                explanation_output += f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Why it matters:</em> This phrase may contain strong or emotionally charged language.<br>"

    return explanation_output


###############################################################################33
###############################################################################33
###############################################################################33
###############################################################################33
###############################################################################33
###############################################################################33
###############################################################################33


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            article = data.get("article", "")
            model_type = data.get("model_type", "bert").lower()
        else:
            article = request.form.get("article", "")
            model_type = request.form.get("model_type", "bert").lower()

        if not article:
            return jsonify({"error": "No article provided"}), 400

        if model_type == "bilstm":
            prediction_value = predict_bilstm(article)
            prediction_value = float(prediction_value)
            label = "Real" if prediction_value > 0.5 else "Fake"
            explanation = explain_bilstm_with_sentences_and_phrases(article)
            print(
                f"Score: {prediction_value} - Label: {label} - Explanation: {explanation}"
            )

        elif model_type == "bert":
            prediction_value = predict_distilbert(article)
            prediction_value = float(prediction_value)
            label = "Real" if prediction_value > 0.5 else "Fake"
            explanation = explain_distilbert_with_phrases(article, truncate=True)
            print(
                f"Score: {prediction_value} - Label: {label} - Explanation: {explanation}"
            )

        else:
            return jsonify({"error": f"Unknown model_type: {model_type}"}), 400

        explanation_str = str(explanation) if explanation else "Explanation missing."

        return jsonify(
            {
                "prediction": float(prediction_value),
                "label": label,
                "Decision Explanation": explanation_str,
            }
        )

    except Exception as e:
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
