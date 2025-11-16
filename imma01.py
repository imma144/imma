from flask import Flask, render_template, request
import re
import logging
import unicodedata
import os
import json
import difflib
import random

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_text(s):
    """Normalize input text for simpler matching:
    - Unicode NFKC normalization
    - Remove Arabic diacritics and tatweel
    - Lowercase and collapse punctuation to spaces
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # remove Arabic diacritics and elongation character
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0640]", "", s)
    s = s.lower()
    # replace non-word (except Arabic range) with space
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------
#   Emotion & Greeting Data
# ---------------------------

EMOTION_RESPONSES = {
    "سعيد": "😊 يبدو أنك  في حالة جيدةاليوم ماذا تحتاج لزيادة سعادتك!",
    "فرح": "😄 يومك جميل ومليء بالسعادة ما سبب هذه الفرحة أخبرني!",
    "حزين": "😢 يبدو أنك حزين... لا تقلق كل شيئ سكون جيدا هل تريد التحدث عن ذلك؟",
    "غاضب": "😠 تشعر بالغضب... خذ نفساً عميقاً لأن لا شيئ يستحق أن تغضب من أجله.",
    "متوتر":"😰 أشعر أنك متوتر... ما الذي جعلك في هذا الحال أنا هنا معك.",
    "متحمس": "🤩 يا لها من طاقة! أنت متحمس جداً جعلني هذا أتحمس معك!",
    "ضايق": "😕 يبدو أنك منزعج قليلاً.",
    "ملل": "😐 تشعر بالملل... لنحاول تغيير الجو!",
}

# Basic polarity mapping for emotions (1 = positive, -1 = negative, 0 = neutral)
EMOTION_POLARITY = {
    "سعيد": 1,
    "مبتهج": 1,
    "فرح": 1,
    "مسرور": 1,
    "متحمس": 1,
    "نشاط": 1,
    "حزين": -1,
    "مكتئب": -1,
    "غاضب": -1,
    "معصب": -1,
    "متوتر": -1,
    "قلق": -1,
    "ضايق": -1,
    "منزعج": -1,
    "ملل": -1,
    "زهقان": -1,
}

# Arabic simple stemmer: remove common prefixes/suffixes
_ARABIC_PREFIXES = ("ال", "و", "ف", "ب", "ل", "ك")
_ARABIC_SUFFIXES = ("ه", "ها", "ان", "ون", "ين", "ات", "ة")

def simple_stem(word):
    # remove prefixes
    for p in _ARABIC_PREFIXES:
        if word.startswith(p) and len(word) > len(p) + 1:
            word = word[len(p):]
            break
    # remove suffixes
    for s in _ARABIC_SUFFIXES:
        if word.endswith(s) and len(word) > len(s) + 1:
            word = word[:-len(s)]
            break
    return word


def detect_sentiment(message):
    """Hybrid sentiment detection:
    1. Try to use a transformers sentiment model if available (may require internet to download the model).
    2. Fallback to a simple lexicon scoring using EMOTION_POLARITY and fuzzy matching.
    Returns: (label, score) where label in ('positive','neutral','negative') and score is float.
    """
    text = message or ""
    # 1) Try transformers pipeline if installed
    try:
        from transformers import pipeline # type: ignore
        # use a multilingual sentiment model as default (may download on first run)
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        nlp = pipeline("sentiment-analysis", model=model_name)
        out = nlp(text[:512])
        if out and isinstance(out, list):
            lab = out[0].get("label", "")
            # label examples: '1 star' .. '5 stars' (nlptown)
            if "1" in lab or "2" in lab:
                return ("negative",  -1.0)
            elif "3" in lab:
                return ("neutral", 0.0)
            else:
                return ("positive", 1.0)
    except Exception:
        # transformers not available or failed; fall back to lexicon
        logger.debug("transformers sentiment unavailable, using lexicon fallback")
        pass

    # 2) Lexicon fallback: normalize, split and score
    norm = normalize_text(text)
    words = [simple_stem(w) for w in norm.split()]
    norm_emotions = {simple_stem(normalize_text(k)): v for k, v in EMOTION_RESPONSES.items()}
    norm_polarity = {simple_stem(normalize_text(k)): EMOTION_POLARITY.get(k, 0) for k in EMOTION_RESPONSES.keys()}

    score = 0.0
    matches = 0
    for w in words:
        # exact match
        if w in norm_polarity:
            score += norm_polarity[w]
            matches += 1
        else:
            close = difflib.get_close_matches(w, list(norm_polarity.keys()), n=1, cutoff=0.78)
            if close:
                score += norm_polarity[close[0]]
                matches += 1

    if matches == 0:
        return ("neutral", 0.0)
    avg = score / max(1, matches)
    if avg > 0.2:
        return ("positive", round(avg, 2))
    if avg < -0.2:
        return ("negative", round(avg, 2))
    return ("neutral", round(avg, 2))

GREETING_RESPONSES = {
    "سلام": "👋 وعليكم السلام! كيف يمكنني مساعدتك؟",
    "مرحبا": "👋 مرحباً! أنا إيما، يسعدني التحدث معك!",
    "اهلا": "👋 أهلاً! كيف تشعر اليوم؟",
    "هاي": "👋 هاي! كيف يومك؟",
    "من": "🤖 أنا إيما، محللة المشاعر الخاصة بك. أخبرني ما تشعر به!",
    "عن": "🤖 أنا هنا لأسمعك وأفهم مشاعرك 💛",
}

SPECIAL_RESPONSES = {
    "?": "🤔 سؤال جميل… هل تريد توضيح المزيد؟",
    "!": "😲 يبدو أنك متأثر جداً بما قلت!",
}

FUNNY_RESPONSES = [
    "😂 حسناً، هذا يكفي لتحسين يومك!",
    "🤣 ضحكت من هذا! أنت تحتاج إلى كوميديان في حياتك!",
    "😄 هذا جعلني أبتسم! شكراً لتحسين يومي أيضاً!",
    "🎭 أنت ممثل كوميديا حقيقي! استمر في هذا!",
    "😆 هذا أفضل شيء سمعته اليوم!",
    "🎪 يالا يا مهرج، استمر!",
    "😅 أعتقد أن لدينا متابع حقيقي للفكاهة هنا!",
    "🤭 هذا سري... لكنك مضحك حقاً!",
    "🌟 أنت نجم! تستحق جائزة أفضل نكتة!",
    "🎬 هذا يستحق أن يكون فيلم كوميديا!",
    "😏 أنا محترفة، لكنك أضحكت حتى إيما!",
    "🎯 أنت محترف في تحسين المزاج!",
    "🔥 يا إلهي! هذا حار جداً! 🌶️",
    "💫 ستاركز! أنت تلمع من السعادة!",
    "🎉 حفلة هنا! أين الموسيقى؟ 🎵",
    "🦸 أنت بطل! لا تنسى عباءتك الخارقة!",
    "🚀 أنت تحلق عالياً! لا تنسى أن تنزل! 😄",
    "👑 ملك المزاح! تاج من ذهب عليك!",
    "💎 أنت ماسة! أما أنا فألماس زيركون... حسناً، على الأقل أنا أتلألأ!",
    "🌈 أنت تصنع قوس قزح بالفرح! رائع!",
    "🎪 سيرك الحياة يبدأ معك!",
    "🤖 آآآآه، مشاعرك جعلت بطاريتي تمتلئ!",
    "😎 أنت بارد جداً! أعطني نظارات شمسية أيضاً!",
    "🎸 أنت أسطورة! هل تريد أن تعزف الجيتار؟",
    "🍕 أنت أفضل من البيتزا! (وهذا قول كثير)",
    "🎓 حكيم المزاح الحقيقي! يستحق درجة دكتوراه!",
    "🌸 أنت زهرة جميلة! (لا... نكتة جادة)",
    "⚡ طاقة خارقة كشفت للتو!",
    "🍀 أنت محظوظ! يا لا! أنت نحن حظًا!",
    "🎊 حفل بمناسبة وجودك!",
]

# Try to load external responses (data/responses.json) to allow easy extension
RESPONSES_PATH = os.path.join(os.path.dirname(__file__), "data", "responses.json")
EXAMPLE_PHRASES = []

def load_external_responses():
    global EMOTION_RESPONSES, GREETING_RESPONSES, SPECIAL_RESPONSES, EXAMPLE_PHRASES
    try:
        if os.path.exists(RESPONSES_PATH):
            with open(RESPONSES_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # update greetings
            if isinstance(data.get("greetings"), dict):
                GREETING_RESPONSES = {**GREETING_RESPONSES, **data.get("greetings")}
            # update emotions
            if isinstance(data.get("emotions"), dict):
                EMOTION_RESPONSES = {**EMOTION_RESPONSES, **data.get("emotions")}
            # update special
            if isinstance(data.get("special"), dict):
                SPECIAL_RESPONSES = {**SPECIAL_RESPONSES, **data.get("special")}
            # examples
            if isinstance(data.get("examples"), list):
                EXAMPLE_PHRASES = data.get("examples")
            logger.info("Loaded external responses from %s", RESPONSES_PATH)
    except Exception:
        logger.exception("Failed to load external responses")


# load on import
load_external_responses()


# ---------------------------
#     Core Analyzer Logic
# ---------------------------
def analyze_emotions(message):
    try:
        text_norm = normalize_text(message)
        words = text_norm.split()

        result_parts = []

        # 1. تحليل التحيات (باستخدام النص المُطَبَّع)
        for greeting, response in GREETING_RESPONSES.items():
            if normalize_text(greeting) in text_norm:
                result_parts.append(response)
                break

        # 2. اكتشاف المشاعر باستخدام تحليل المشاعر الهجين
        sentiment_label, sentiment_score = detect_sentiment(message)
        
        # 3. ردود بناءً على تحليل المشاعر (نتائج أكثر دقة)
        if sentiment_label == "positive":
            positive_responses = [
                "😊 يبدو أنك في حالة جيدة! هذا رائع!",
                "😄 أرى أن لديك طاقة إيجابية! استمر في هذا!",
                "🤩 يا لها من طاقة رائعة! أنت متفائل حقاً!",
                "😊 أشعر بإيجابيتك! هذا معدي!",
                "🌟 مزاجك ممتاز اليوم! ماذا يجعلك سعيداً؟",
                "💫 انتقلت لي سعادتك! شكراً لك!",
            ]
            result_parts.append(random.choice(positive_responses))
        elif sentiment_label == "negative":
            negative_responses = [
                "😔 أرى أنك تمر بوقت صعب... أنا هنا لك.",
                "😞 يبدو أن هناك شيئاً ما يزعجك... هل تريد الحديث عنه؟",
                "💔 أشعر بحزنك... لا تقلق، هذا سيمر.",
                "😢 الحياة ليست دائماً سهلة... لكنك قوي!",
                "🌧️ كل الأيام المظلمة لها غروب شمس جميل.",
                "💪 أنت أقوى مما تعتقد! ستتجاوز هذا!",
            ]
            result_parts.append(random.choice(negative_responses))
        else:  # neutral
            neutral_responses = [
                "🤔 أنت في حالة محايدة... هل كل شيء بخير؟",
                "😐 يبدو أن يومك عادي... ماذا يمكنني أن أفعل لتحسينه؟",
                "💭 أشعر بك في نقطة التوازن...",
                "🙂 حسناً، ما رأيك نجعل اليوم أفضل؟",
            ]
            result_parts.append(random.choice(neutral_responses))

        # 4. تحليل الكلمات المحددة مسبقاً (للدقة الإضافية)
        norm_emotions = {normalize_text(k): v for k, v in EMOTION_RESPONSES.items()}
        matched = set()
        for word in words:
            if word in norm_emotions:
                resp = norm_emotions[word]
                if resp not in matched:
                    result_parts.append(resp)
                    matched.add(resp)
            else:
                # fuzzy match against emotion keys
                close = difflib.get_close_matches(word, list(norm_emotions.keys()), n=1, cutoff=0.78)
                if close:
                    resp = norm_emotions[close[0]]
                    if resp not in matched:
                        result_parts.append(resp)
                        matched.add(resp)

        # 5. رموز خاصة (نستخدم النص الأصلي كي لا نخسر علامات الترقيم)
        if "؟" in message or "?" in message:
            result_parts.append(SPECIAL_RESPONSES["?"])

        if "!" in message:
            result_parts.append(SPECIAL_RESPONSES["!"])

        # 6. طول النص
        if len(words) > 10:
            result_parts.append("📝 رسالتك طويلة… يبدو أنك تفكر كثيراً.")
        elif len(words) <= 2:
            if message.strip() == "":
                result_parts.append("💬 الرجاء كتابة رسالة حتى أساعدك.")
            else:
                result_parts.append("💭 رسالتك قصيرة… قل لي المزيد!")

        # 7. إضافة رد طريف عشوائي (40% احتمالية)
        if result_parts and random.random() > 0.6:
            funny_response = random.choice(FUNNY_RESPONSES)
            result_parts.append("😄 " + funny_response)

        # 8. لا شيء مفهوم
        if not result_parts:
            default_responses = [
                "🤔 فهمت… لكن كيف يجعلك هذا تشعر؟",
                "💬 يبدو أنك تقول شيئاً مهماً... أخبرني المزيد!",
                "🎯 أنت تجعلني مهتمة! استمر من فضلك!",
            ]
            result_parts.append(random.choice(default_responses))

        return " ".join(result_parts)
    except Exception:
        logger.exception("خطأ أثناء تحليل الرسالة")
        return "عذراً، حدث خطأ أثناء تحليل الرسالة. حاول مرة أخرى لاحقاً."


# ---------------------------
#       Flask Route
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    sentiment_label = None
    sentiment_score = None
    examples = EXAMPLE_PHRASES

    if request.method == "POST":
        input_text = request.form.get("user_text", "")
        result = analyze_emotions(input_text)
        sentiment_label, sentiment_score = detect_sentiment(input_text)

    return render_template("mood_analyzer.html",
                           result=result,
                           input_text=input_text,
                           sentiment_label=sentiment_label,
                           sentiment_score=sentiment_score,
                           examples=examples)


# ---------------------------
#      Run Server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)