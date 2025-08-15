import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Career Recommender", page_icon="", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .main-header p {
        color: white !important;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .stSlider > div > div > div > div {
        background-color: #FF6B6B;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        color: white;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<div class="main-header"><h1>🚀 Career Path Recommender</h1><p>Answer these questions to get personalized career recommendations.</p></div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('career_recommender_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found! Please run the training notebook first.")
        return None

# Load model
data = load_model()
if data is None:
    st.stop()

model, feature_cols, profiles = data['model'], data['feature_cols'], data['profiles']

# Questionnaire scoring
def score_questionnaire(responses):
    work_values_resp = responses[:6]
    riasec_resp = responses[6:36]
    work_styles_resp = responses[36:50]
    
    wv_scores = {
        'Achievement': np.mean([work_values_resp[0], work_values_resp[1]]) / 5.0,
        'Working Conditions': np.mean([work_values_resp[2], work_values_resp[3]]) / 5.0,
        'Relationships': np.mean([work_values_resp[4], work_values_resp[5]]) / 5.0
    }
    
    riasec_scores = {}
    categories = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    for i, category in enumerate(categories):
        start_idx = 6 + (i * 5)
        end_idx = start_idx + 5
        riasec_scores[category] = np.mean(riasec_resp[start_idx:end_idx]) / 5.0
    
    ws_scores = {
        'Achievement/Effort': np.mean(work_styles_resp[0:3]) / 5.0,
        'Leadership': work_styles_resp[3] / 5.0,
        'Cooperation': np.mean(work_styles_resp[4:6]) / 5.0,
        'Stress Tolerance': np.mean(work_styles_resp[6:8]) / 5.0,
        'Dependability': np.mean(work_styles_resp[8:10]) / 5.0,
        'Adaptability': np.mean(work_styles_resp[10:12]) / 5.0,
        'Innovation': work_styles_resp[12] / 5.0,
        'Analytical Thinking': work_styles_resp[13] / 5.0
    }
    
    return {**wv_scores, **riasec_scores, **ws_scores}

# Get recommendations
def get_career_recommendations(user_profile, model, feature_cols, profiles, top_k=10):
    # Find column names
    soc_col = None
    title_col = None
    for col in profiles.columns:
        if 'soc' in col.lower() or 'code' in col.lower():
            soc_col = col
        if 'title' in col.lower():
            title_col = col
    
    if soc_col is None or title_col is None:
        return None
    
    user_vector = np.array([user_profile.get(col, 0) for col in feature_cols])
    predicted_profile = model.predict(user_vector.reshape(1, -1))[0]
    
    occupation_features = profiles[feature_cols].values
    similarities = cosine_similarity(occupation_features, predicted_profile.reshape(1, -1)).ravel()
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    recommendations = profiles.iloc[top_indices][[soc_col, title_col]].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    return recommendations.reset_index(drop=True)

# Questions array - exactly 50 questions
questions = [
    "Being able to use your strongest abilities and see results",
    "Getting recognition for the work you do", 
    "Having good working conditions",
    "Getting support from management and coworkers",
    "Having co-workers who are easy to get along with",
    "Being able to work on your own",
    "Working with your hands",
    "Building or repairing things",
    "Operating machinery",
    "Working outdoors",
    "Working with animals",
    "Solving complex problems",
    "Conducting research",
    "Analyzing data",
    "Working with computers",
    "Learning new things",
    "Creating artwork",
    "Writing stories or poetry",
    "Designing things",
    "Performing music or drama",
    "Expressing yourself creatively",
    "Teaching or training others",
    "Helping people with problems",
    "Working in teams",
    "Caring for others",
    "Public speaking",
    "Leading or managing others",
    "Selling products or ideas",
    "Making business decisions",
    "Starting your own business",
    "Negotiating or persuading",
    "Organizing information",
    "Following detailed procedures",
    "Working with numbers",
    "Keeping accurate records",
    "Working in structured environments",
    "Setting high goals and working hard to achieve them",
    "Persisting in completing tasks even when difficult",
    "Taking initiative to start projects",
    "Enjoying taking charge and leading others",
    "Working well with people and enjoying teamwork",
    "Being concerned about the well-being of others",
    "Adapting easily to new situations and changes",
    "Paying close attention to details",
    "Being counted on to complete work reliably",
    "Maintaining high ethical standards in work",
    "Working independently without supervision",
    "Handling stressful situations well",
    "Maintaining self-control even in difficult circumstances",
    "Being flexible and open to new ideas"
]

# Verify question count
TOTAL_QUESTIONS = len(questions)
print(f"Total questions: {TOTAL_QUESTIONS}")

# Initialize responses with correct length
if 'responses' not in st.session_state:
    st.session_state.responses = [3] * TOTAL_QUESTIONS

# Display questions with better styling
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
    <h3 style="color: white; text-align: center; margin: 0 0 1rem 0;">📋 Assessment Questions</h3>
    <p style="color: white; text-align: center; margin: 0; opacity: 0.9;">
        Rate each statement from 1 (Strongly Disagree/Not Important) to 5 (Strongly Agree/Extremely Important)
    </p>
</div>
""", unsafe_allow_html=True)

# Display questions in left column
for i in range(TOTAL_QUESTIONS):
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #FF6B6B;">
        <strong style="color: #FF6B6B;">Q{i+1}:</strong> {questions[i]}
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.responses[i] = st.slider(
        f"Rating:",
        min_value=1,
        max_value=5,
        value=st.session_state.responses[i],
        key=f"q_{i}",
        label_visibility="collapsed"
    )

# Submit button with better styling
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h3 style="color: #FF6B6B; margin-bottom: 1rem;">Ready to get your recommendations?</h3>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Get Career Recommendations", type="primary", key="submit_button"):
    user_profile = score_questionnaire(st.session_state.responses)
    recommendations = get_career_recommendations(user_profile, model, feature_cols, profiles)
    
    if recommendations is not None:
        st.session_state.recommendations = recommendations
        
        # Success message with styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50, #45a049); padding: 1.5rem; border-radius: 10px; margin: 2rem 0; text-align: center;">
            <h3 style="color: white; margin: 0;">🎉 Assessment Completed!</h3>
            <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Scroll down to see your personalized career recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show results with better styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
            <h3 style="color: white; text-align: center; margin: 0 0 1rem 0;">🌟 Your Top Career Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Style the dataframe
        st.dataframe(
            recommendations, 
            hide_index=True,
            use_container_width=True
        )
        
        # Download button
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name="career_recommendations.csv",
            mime="text/csv",
            key="download_button"
        )
    else:
        st.error("Error generating recommendations. Please try again.")