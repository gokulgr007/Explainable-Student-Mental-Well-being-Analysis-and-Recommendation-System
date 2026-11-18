
import streamlit as st
import pandas as pd
import joblib
import nltk

from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Wellbeing Analyzer",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# 2. NLP PREPROCESSING
# ==========================================

@st.cache_resource
def load_nlp_resources():

    for package in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        nltk.download(package, quiet=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nlp_resources()


def preprocess_text(text):

    tokens = word_tokenize(text.lower())

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return ' '.join(tokens)


# ==========================================
# 3. LOAD MODEL AND NLP ARTIFACTS
# ==========================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("artifacts/ml_model.pkl")
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    recommendation_df = joblib.load("artifacts/recommendations.pkl")

    recommendation_matrix = tfidf.transform(
        recommendation_df['cleaned_text']
    )

    return model, tfidf, recommendation_df, recommendation_matrix


model, tfidf, recommendation_df, recommendation_matrix = load_artifacts()


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.title("🧠 Student Wellbeing Analyzer")

st.write(
    "Analyze student wellbeing patterns and "
    "receive personalized general wellbeing suggestions."
)

st.divider()


# ==========================================
# 5. STUDENT INPUT FORM
# ==========================================

st.subheader("Student Information")

with st.form("student_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input("Age", 18, 24, 22)

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        academic_level = st.selectbox(
            "Academic Level",
            ["Graduate", "Undergraduate", "High School"]
        )

        stress = st.selectbox(
            "Stress Level",
            ["Low", "Medium", "High", "Very High"]
        )

    with col2:

        study_hours = st.number_input(
            "Study Hours", 0.5, 8.3, 5.0
        )

        sleep_hours = st.number_input(
            "Sleep Hours Per Night", 3.6, 9.9, 7.0
        )

        activity_hours = st.number_input(
            "Physical Activity Hours", 0.0, 4.1, 1.5
        )

        usage_hours = st.number_input(
            "Daily Social Media Usage Hours", 1.1, 8.8, 4.0
        )

    with col3:

        unlocks = st.number_input(
            "Daily Unlocks", 62, 273, 150
        )

        platform = st.selectbox(
            "Most Used Platform",
            [
                "Facebook", "Instagram", "TikTok", "LinkedIn",
                "Snapchat", "WhatsApp", "LINE", "Twitter",
                "VKontakte", "YouTube", "WeChat", "KakaoTalk"
            ]
        )

        purpose = st.selectbox(
            "Purpose Of Use",
            ["Networking", "Entertainment", "Education", "News"]
        )

        country = st.selectbox(
            "Country",
            [
                "Mexico", "Other", "France", "Canada", "USA",
                "India", "Australia", "UK", "Turkey", "Germany"
            ]
        )

    st.divider()

    submitted = st.form_submit_button(
        "Analyze My Wellbeing",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 6. ML PREDICTION AND NLP RECOMMENDATIONS
# ==========================================

if submitted:

    student = {

        "Study_Hours": study_hours,
        "Stress_Level": stress,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "grouped_country": country,
        "Age": age,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Daily_Unlocks": unlocks,
        "Physical_Activity_Hours": activity_hours,
        "Sleep_Hours_Per_Night": sleep_hours

    }

    # --------------------------------------
    # ML PREDICTION
    # --------------------------------------

    student_df = pd.DataFrame([student])

    predicted_score = model.predict(student_df)[0]

    st.success("Analysis completed successfully!")

    st.subheader("Predicted Mental Health Score")

    st.metric(
        "Predicted Score",
        f"{predicted_score:.2f} / 10"
    )

    st.divider()


    # --------------------------------------
    # IDENTIFY IMPROVEMENT AREAS
    # --------------------------------------

    areas = []

    if sleep_hours < 7:
        areas.append(
            'insufficient sleep poor sleep schedule bedtime routine'
        )

    if stress in ['High', 'Very High']:
        areas.append(
            'high stress anxiety relaxation breathing exercises'
        )

    if usage_hours > 5:
        areas.append(
            'excessive social media screen time reduce phone usage'
        )

    if activity_hours < 1:
        areas.append(
            'low physical activity exercise walking yoga'
        )

    if study_hours > 8:
        areas.append(
            'academic workload study balance time management'
        )


    # --------------------------------------
    # NLP RECOMMENDATIONS
    # --------------------------------------

    st.subheader("Personalized Wellbeing Suggestions")

    if areas:

        for area in areas:

            cleaned_query = preprocess_text(area)

            query_vector = tfidf.transform([cleaned_query])

            similarity_scores = cosine_similarity(
                query_vector,
                recommendation_matrix
            )

            best_index = similarity_scores.argmax()

            best_score = similarity_scores[0][best_index]

            if best_score > 0:

                result = recommendation_df.iloc[best_index]

                st.markdown(f"### {result['category']}")

                st.write(result['text'])

                st.divider()

            else:
                st.info("No relevant recommendation found.")

    else:

        st.success(
            "No improvement areas were flagged based on the current rules!"
        )

    st.caption(
        "These suggestions provide general wellbeing support "
        "and are not a medical diagnosis or treatment."
    )
