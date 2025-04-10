import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="MindMitra", page_icon="💬", layout="centered")
st.title("🧠 MindMitra: Your Mental Health Therapist")
st.write("This analyzer detects mental health issues from user input and suggests resources. (यह चैटबॉट उपयोगकर्ता इनपुट से मानसिक स्वास्थ्य समस्याओं का पता लगाता है और संसाधन सुझाता है।)")
st.write("💙 **Talk to me, and I'll try to understand how you're feeling.**")

# Use caching for the model to improve performance and prevent timeouts
@st.cache_resource
def load_emotion_classifier():
    try:
        # Use a smaller distilled model with direct pipeline loading
        return pipeline("text-classification", 
                       model="j-hartmann/emotion-english-distilroberta-base",return_all_scores=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Mental health categories and keywords
MENTAL_HEALTH_CATEGORIES = {
    "Depression": ["hopeless", "useless", "worthless", "empty", "lost", "numb", "fatigued", "drained", "unmotivated", "exhausted"],
    "Anxiety": ["worried", "nervous", "panic", "overwhelmed", "restless", "tense", "uneasy", "overthinking", "dizzy", "racing thoughts"],
    "Sadness": ["lonely", "heartbroken", "disappointed", "crying", "upset", "sorrow", "unhappy", "down", "missing", "abandoned"],
    "Loneliness": ["alone", "isolated", "ignored", "misunderstood", "disconnected", "forgotten", "unwanted", "invisible", "left out", "empty"],
    "Stress": ["pressure", "overwhelmed", "exhausted", "burnout", "headache", "suffocating", "responsibilities", "workload", "breaking point", "tense"],
    "Anger": ["furious", "betrayed", "irritated", "frustrated", "rage", "unfair", "offended", "resentful", "yelling", "aggressive"],
    "Burnout": ["drained", "exhausted", "done", "can't anymore", "overworked", "detached", "unmotivated", "helpless", "fatigue", "numb"],
    "Self-Harm/Suicidal Thoughts": ["die", "end it", "suicide", "cut", "harm", "death", "disappear", "no point", "nothing left", "burden"]
}

MENTAL_HEALTH_RESOURCES = {
    "Depression": ("👨🏻‍⚕️ Dr. Rajesh Verma - +91 98765 43210", "https://www.mentalhealthindia.net/"),
    "Anxiety": ("👨🏻‍⚕️ Dr. Pooja Sharma - +91 91234 56789", "https://www.mentalhealthindia.net/"),
    "Sadness": ("👨🏻‍⚕️ Dr. Vikram Patel - +91 88990 11223", "https://www.mentalhealthindia.net/"),
    "Loneliness": ("👨🏻‍⚕️ Dr. Neha Gupta - +91 99887 77665", "https://www.mentalhealthindia.net/"),
    "Stress": ("👨🏻‍⚕️ Dr. Arjun Mehta - +91 90909 80808", "https://www.mentalhealthindia.net/"),
    "Anger": ("👨🏻‍⚕️ Dr. Anjali Deshmukh - +91 81234 56789", "https://www.mentalhealthindia.net/"),
    "Burnout": ("👨🏻‍⚕️ Dr. Ramesh Iyer - +91 92345 67890", "https://www.mentalhealthindia.net/"),
    "Self-Harm/Suicidal Thoughts": ("Vandrevala Foundation Helpline - 1860 266 2345", "https://www.snehi.org/")
}

# Input box
user_input = st.text_area("💬Type your thoughts here: (कृपया अपनी भावनाएं यहाँ लिखें:)")

# Show a loading indicator while processing
if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing your thoughts..."):
            # Load the model (cached)
            classifier = load_emotion_classifier()
            
            if classifier:
                try:
                    # Get emotion predictions - this returns a list of dictionaries
                    emotions = classifier(user_input)
                    
                    # Debug info
                    st.write(f"Debug - emotions type: {type(emotions)}")
                    
                    # Fix: Make sure we handle the emotion output format correctly
                    emotion_scores = {}
                    if isinstance(emotions, list):
                        # If it's a list with one item (which is a dict)
                        if len(emotions) > 0 and isinstance(emotions[0], dict):
                            emotion_scores = {emotions[0]['label']: emotions[0]['score']}
                            # If there are multiple labels
                            if 'label' in emotions[0] and isinstance(emotions[0]['label'], list):
                                for i, label in enumerate(emotions[0]['label']):
                                    emotion_scores[label] = emotions[0]['score'][i]
                    
                    # Detect issues based on keywords
                    detected_issues = []
                    help_resources = []

                    for category, keywords in MENTAL_HEALTH_CATEGORIES.items():
                        if any(word in user_input.lower() for word in keywords):
                            detected_issues.append(category)
                            if category in MENTAL_HEALTH_RESOURCES:
                                help_resources.append(MENTAL_HEALTH_RESOURCES[category])

                    issue_label = ", ".join(detected_issues) if detected_issues else "No clear issue detected (कोई स्पष्ट समस्या नहीं मिली )👍"

                    st.subheader("📝 Mental Health Classification (मानसिक स्वास्थ्य वर्गीकरण)")
                    st.write(f"**🔹 Detected Issue (पहचानी गई समस्या):** {issue_label}")

                    if help_resources:
                        st.subheader("📌 Suggested Resources (सुझाए गए संसाधन):")
                        for contact, link in help_resources:
                            st.write(f"📞 **{contact}**")
                            st.markdown(f"🔗 [सहायता लिंक (Help Link)]({link})")

                    # Show emotion scores if we have any
                    if emotion_scores:
                        st.subheader("📊 Emotion Analysis (भावना विश्लेषण )")
                        st.bar_chart(emotion_scores)
                    else:
                        st.warning("Couldn't analyze emotions from the text.")
                        
                except Exception as e:
                    st.error(f"Error analyzing text: {e}")
                    st.write("Debug info:")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.error("Could not load the emotion classification model. Please try again later.")
    else:
        st.warning("Please enter some text for analysis. (कृपया विश्लेषण के लिए कुछ पाठ दर्ज करें।)")

st.markdown("---")
st.write("💡 *This chatbot is for informational purposes only. If you are in crisis, please seek professional help.*")
