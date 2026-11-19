import streamlit as st
import pandas as pd
import joblib
import nltk
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


st.set_page_config(
    page_title="Student Wellbeing Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Clean up top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Hero Score Card */
    .score-card {
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .score-value {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.4rem 0;
    }
    .score-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.9;
    }
    /* Factor Pills */
    .factor-box {
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 500;
    }
    .factor-pos {
        background-color: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #10b981;
    }
    .factor-neg {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
    }
    /* Recommendation Cards */
    .rec-card {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_LABELS = {
    'Study_Hours': '📚 Daily Study Hours',
    'Avg_Daily_Usage_Hours': '📱 Social Media Screen Time',
    'Daily_Unlocks': '🔓 Daily Phone Unlocks',
    'Physical_Activity_Hours': '🏃 Physical Activity',
    'Sleep_Hours_Per_Night': '😴 Sleep Duration',
    'Stress_Level': '⚡ Perceived Stress Level'
}


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
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "models")

    model = joblib.load(os.path.join(model_dir, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    recommendation_df = joblib.load(os.path.join(model_dir, "recommendations.pkl"))

    recommendation_matrix = tfidf.transform(recommendation_df['cleaned_text'])

    rf_model = model.named_steps['random forest']
    preprocessor = model.named_steps['preprocessor']
    explainer = shap.TreeExplainer(rf_model)

    feature_names = [
        'Study_Hours', 'Age', 'Avg_Daily_Usage_Hours',
        'Daily_Unlocks', 'Physical_Activity_Hours',
        'Sleep_Hours_Per_Night', 'Stress_Level'
    ]

    onehot_encoder = (
        preprocessor
        .named_transformers_['Nominal']
        .named_steps['encode']
    )
    feature_names.extend(onehot_encoder.get_feature_names_out())

    return (
        model, tfidf, recommendation_df,
        recommendation_matrix, preprocessor,
        explainer, feature_names
    )


(
    model, tfidf, recommendation_df,
    recommendation_matrix, preprocessor,
    explainer, feature_names
) = load_models()


with st.sidebar:
    st.header("📋 Student Profile")
    st.caption("Adjust the parameters below and run the analyzer.")

    with st.form("student_form"):
        st.markdown("**👤 Demographics**")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age", 18, 24, 22)
        with col_b:
            gender = st.selectbox("Gender", ["Male", "Female"])

        academic_level = st.selectbox(
            "Academic Level",
            ["Undergraduate", "Graduate", "High School"]
        )
        country = st.selectbox(
            "Country",
            ["India", "USA", "UK", "Canada", "Australia", "Germany", "France", "Mexico", "Turkey", "Other"]
        )

        st.divider()
        st.markdown("**⏳ Daily Habits & Wellbeing**")
        sleep_hours = st.slider("😴 Sleep (hrs/night)", 3.6, 9.9, 7.0, 0.1)
        study_hours = st.slider("📚 Study (hrs/day)", 0.5, 8.3, 5.0, 0.1)
        activity_hours = st.slider("🏃 Physical Activity (hrs/day)", 0.0, 4.1, 1.5, 0.1)
        stress = st.select_slider(
            "⚡ Stress Level",
            options=["Low", "Medium", "High", "Very High"],
            value="Medium"
        )

        st.divider()
        st.markdown("**📱 Digital Footprint**")
        usage_hours = st.slider("Screen Time (hrs/day)", 1.1, 8.8, 4.0, 0.1)
        unlocks = st.slider("Daily Phone Unlocks", 62, 273, 150)

        col_c, col_d = st.columns(2)
        with col_c:
            platform = st.selectbox(
                "Top Platform",
                [
                    "Instagram", "YouTube", "WhatsApp", "TikTok",
                    "LinkedIn", "Snapchat", "Facebook", "Twitter",
                    "LINE", "VKontakte", "WeChat", "KakaoTalk"
                ]
            )
        with col_d:
            purpose = st.selectbox(
                "Primary Use",
                ["Entertainment", "Networking", "Education", "News"]
            )

        submitted = st.form_submit_button(
            "✨ Analyze My Wellbeing",
            type="primary",
            use_container_width=True
        )



st.title("🧠 Student Wellbeing & Digital Habit Analyzer")
st.markdown(
    "Evaluate how daily routines, academic load, and screen habits shape your "
    "mental wellbeing—powered by **Explainable AI (SHAP)** and **NLP recommendations**."
)

if not submitted:
    st.info("👈 **Get Started:** Configure the student profile in the left sidebar and click **Analyze My Wellbeing**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 🎯 1. Predict Score")
            st.caption("Estimates an overall mental wellbeing index out of 10 from lifestyle & screen-time metrics.")
    with c2:
        with st.container(border=True):
            st.markdown("### 🔍 2. Explain Drivers")
            st.caption("Uses SHAP TreeExplainer to reveal exactly which habits lifted or dragged down your score.")
    with c3:
        with st.container(border=True):
            st.markdown("### 💡 3. Action Plan")
            st.caption("Matches your flagged risk areas with tailored habits using TF-IDF cosine similarity.")
    st.stop()



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

student_df = pd.DataFrame([student])
predicted_score = float(model.predict(student_df)[0])

if predicted_score >= 7.0:
    bg_gradient = "linear-gradient(135deg, #059669 0%, #10b981 100%)"
    status_label = "Thriving Wellbeing"
elif predicted_score >= 5.0:
    bg_gradient = "linear-gradient(135deg, #d97706 0%, #f59e0b 100%)"
    status_label = "Moderate Balance"
else:
    bg_gradient = "linear-gradient(135deg, #dc2626 0%, #ef4444 100%)"
    status_label = "Needs Attention"

hero_col, summary_col = st.columns([1.1, 1.9], gap="large")

with hero_col:
    st.markdown(f"""
        <div class="score-card" style="background: {bg_gradient};">
            <div class="score-label">Predicted Wellbeing Index</div>
            <div class="score-value">{predicted_score:.2f} <span style="font-size:1.5rem;font-weight:400;">/ 10</span></div>
            <div style="font-size: 1.05rem; font-weight: 600;">Status: {status_label}</div>
        </div>
    """, unsafe_allow_html=True)
    st.progress(min(max(predicted_score / 10.0, 0.0), 1.0))

with summary_col:
    st.subheader("Daily Habit Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "😴 Sleep",
        f"{sleep_hours}h",
        delta="Optimal" if sleep_hours >= 7 else "Low (<7h)",
        delta_color="normal" if sleep_hours >= 7 else "inverse"
    )
    m2.metric(
        "📱 Screen Time",
        f"{usage_hours}h",
        delta="High (>5h)" if usage_hours > 5 else "Balanced",
        delta_color="inverse" if usage_hours > 5 else "normal"
    )
    m3.metric(
        "🏃 Activity",
        f"{activity_hours}h",
        delta="Active" if activity_hours >= 1 else "Low (<1h)",
        delta_color="normal" if activity_hours >= 1 else "inverse"
    )
    m4.metric(
        "⚡ Stress",
        stress,
        delta="Elevated" if stress in ["High", "Very High"] else "Manageable",
        delta_color="inverse" if stress in ["High", "Very High"] else "normal"
    )


tab_drivers, tab_recs, tab_waterfall = st.tabs([
    "🎯 Key Habit Drivers",
    "💡 Personalized Action Plan",
    "📊 Detailed SHAP Waterfall"
])

shap_error = None
try:
    student_transformed = preprocessor.transform(student_df)
    if hasattr(student_transformed, "toarray"):
        student_transformed = student_transformed.toarray()

    shap_explanation = explainer(student_transformed)
    contributions = shap_explanation.values[0]

    habit_features = list(FEATURE_LABELS.keys())
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': contributions
    })

    habit_shap = shap_df[shap_df['Feature'].isin(habit_features)].copy()
    habit_shap['Clean_Name'] = habit_shap['Feature'].map(FEATURE_LABELS)
    habit_shap['Impact'] = habit_shap['SHAP_Value'].abs()
    habit_shap = habit_shap.sort_values(by='Impact', ascending=False)
except Exception as e:
    shap_error = e


with tab_drivers:
    if shap_error:
        st.error("Could not compute feature contributions.")
        st.exception(shap_error)
    else:
        st.markdown("##### How Your Daily Habits Shifted Your Score")
        col_pos, col_neg = st.columns(2, gap="large")

        with col_pos:
            st.markdown("###### 🟢 Boosting Your Wellbeing")
            positive = habit_shap[habit_shap['SHAP_Value'] > 0]
            if positive.empty:
                st.caption("No habits currently pushing the score above baseline.")
            else:
                for _, row in positive.iterrows():
                    st.markdown(f"""
                        <div class="factor-box factor-pos">
                            <span>{row['Clean_Name']}</span>
                            <span style="color:#059669; font-weight:700;">+{row['SHAP_Value']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)

        with col_neg:
            st.markdown("###### 🔴 Dragging Your Wellbeing Down")
            negative = habit_shap[habit_shap['SHAP_Value'] < 0]
            if negative.empty:
                st.caption("No habits negatively impacting your score!")
            else:
                for _, row in negative.iterrows():
                    st.markdown(f"""
                        <div class="factor-box factor-neg">
                            <span>{row['Clean_Name']}</span>
                            <span style="color:#dc2626; font-weight:700;">{row['SHAP_Value']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)


with tab_recs:
    areas = []
    if sleep_hours < 7:
        areas.append(('😴 Sleep Hygiene', 'insufficient sleep poor sleep schedule bedtime routine'))
    if stress in ['High', 'Very High']:
        areas.append(('⚡ Stress Management', 'high stress anxiety relaxation breathing exercises'))
    if usage_hours > 5:
        areas.append(('📱 Digital Detox', 'excessive social media screen time reduce phone usage'))
    if activity_hours < 1:
        areas.append(('🏃 Movement & Exercise', 'low physical activity exercise walking yoga'))
    if study_hours > 8:
        areas.append(('📚 Study-Life Balance', 'academic workload study balance time management'))

    if areas:
        st.markdown(f"We identified **{len(areas)} focus area(s)** based on your current daily routine:")
        rec_cols = st.columns(2)
        for idx, (badge_title, query_text) in enumerate(areas):
            cleaned_query = preprocess_text(query_text)
            query_vector = tfidf.transform([cleaned_query])
            similarity_scores = cosine_similarity(query_vector, recommendation_matrix)
            best_index = similarity_scores.argmax()
            best_score = similarity_scores[0][best_index]

            with rec_cols[idx % 2]:
                if best_score > 0:
                    result = recommendation_df.iloc[best_index]
                    st.markdown(f"""
                        <div class="rec-card">
                            <div style="font-size:0.8rem; text-transform:uppercase; color:#6366f1; font-weight:700; margin-bottom:4px;">
                                {badge_title} • Match: {best_score*100:.0f}%
                            </div>
                            <h4 style="margin-top:0; margin-bottom:0.5rem;">{result['category']}</h4>
                            <p style="margin-bottom:0; font-size:0.95rem; opacity:0.9;">{result['text']}</p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.success("🎉 **All core habits look balanced!** Keep maintaining your sleep, activity, and screen-time boundaries.")


with tab_waterfall:
    if shap_error:
        st.error("Waterfall plot unavailable.")
    else:
        st.markdown("##### Exact Mathematical Breakdown (SHAP Waterfall)")
        st.caption(
            "Starts from the dataset's average baseline score E[f(X)] at the bottom and adds each feature's "
            "push to reach your final predicted score f(x) at the top."
        )
        waterfall_explanation = shap.Explanation(
            values=shap_explanation.values[0],
            base_values=np.asarray(shap_explanation.base_values).reshape(-1)[0],
            data=student_transformed[0],
            feature_names=feature_names
        )

        col_plot, _ = st.columns([2, 1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(waterfall_explanation, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

st.divider()
st.caption(
    "Disclaimer: These insights provide general lifestyle and wellbeing support "
    "and are not a substitute for professional medical diagnosis or treatment."
)