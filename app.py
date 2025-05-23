import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

st.set_page_config(page_title="MindMitra", page_icon="💬", layout="centered")

@st.cache_resource
def load_emotion_classifier():
    try:
        return pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base", 
            return_all_scores=True
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


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
    "Depression": ("👨🏻‍⚕️ Dr. Rajesh Verma (डॉ. राजेश वर्मा) - +91 98765 43210", "https://www.mentalhealthindia.net/"),
    "Anxiety": ("👨🏻‍⚕️ Dr. Pooja Sharma (डॉ. पूजा शर्मा) - +91 91234 56789", "https://www.mentalhealthindia.net/"),
    "Sadness": ("👨🏻‍⚕️ Dr. Vikram Patel (डॉ. विक्रम पटेल) - +91 88990 11223", "https://www.mentalhealthindia.net/"),
    "Loneliness": ("👨🏻‍⚕️ Dr. Neha Gupta (डॉ. नेहा गुप्ता) - +91 99887 77665", "https://www.mentalhealthindia.net/"),
    "Stress": ("👨🏻‍⚕️ Dr. Arjun Mehta (डॉ. अर्जुन मेहता) - +91 90909 80808", "https://www.mentalhealthindia.net/"),
    "Anger": ("👨🏻‍⚕️ Dr. Anjali Deshmukh (डॉ. अंजलि देशमुख) - +91 81234 56789", "https://www.mentalhealthindia.net/"),
    "Burnout": ("👨🏻‍⚕️ Dr. Ramesh Iyer (डॉ. रमेश अय्यर) - +91 92345 67890", "https://www.mentalhealthindia.net/"),
    "Self-Harm/Suicidal Thoughts": ("Vandrevala Foundation Helpline (वंद्रेवाला फाउंडेशन हेल्पलाइन) - 1860 266 2345", "https://www.snehi.org/")
}

st.title("🧠 MindMitra: Your Mental Health Therapist")
st.write("This analyzer detects mental health issues from user input and suggests resources. (यह चैटबॉट उपयोगकर्ता इनपुट से मानसिक स्वास्थ्य समस्याओं का पता लगाता है और संसाधन सुझाता है।)")
st.write("💙 **Talk to me, and I'll try to understand how you're feeling.**")

# Input box
user_input = st.text_area("💬Type your thoughts here: (कृपया अपनी भावनाएं यहाँ लिखें:)")
st.write("💙 **Remember, you're not alone. If you're struggling, reach out to someone you trust.**")

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing your thoughts..."):
            detected_issues = []
            help_resources = []

            for category, keywords in MENTAL_HEALTH_CATEGORIES.items():
                if any(word in user_input.lower() for word in keywords):
                    detected_issues.append(category)
                    if category in MENTAL_HEALTH_RESOURCES:
                        help_resources.append(MENTAL_HEALTH_RESOURCES[category])

            issue_label = ", ".join(detected_issues) if detected_issues else "No clear issue detected (कोई स्पष्ट समस्या नहीं मिली )👍"
            
            try:
                classifier = load_emotion_classifier()
                
                if classifier:
                    emotions_result = classifier(user_input)
                    
                    if emotions_result and len(emotions_result) > 0:
                        emotion_scores = {}
                        for emotion_data in emotions_result[0]:
                            if isinstance(emotion_data, dict) and 'label' in emotion_data and 'score' in emotion_data:
                                emotion_scores[emotion_data['label']] = emotion_data['score']

                        # Display results
                        st.subheader("📝 Mental Health Classification (मानसिक स्वास्थ्य वर्गीकरण)")
                        st.write(f"**🔹 Detected Issue (पहचानी गई समस्या):** {issue_label}")

                        if help_resources:
                            st.subheader("📌 Suggested Resources (सुझाए गए संसाधन):")
                            for contact, link in help_resources:
                                st.write(f"📞 **{contact}**")
                                st.markdown(f"🔗 [सहायता लिंक (Help Link)]({link})")

                        if emotion_scores:
                            st.subheader("📊 Emotion Analysis (भावना विश्लेषण )")
                            # Sort emotions by score for better visualization
                            emotion_df = pd.DataFrame({'emotion': list(emotion_scores.keys()), 
                                                     'score': list(emotion_scores.values())})
                            emotion_df = emotion_df.sort_values('score', ascending=False)
                            st.bar_chart(data=emotion_df.set_index('emotion'))
                    else:
                        raise ValueError("Model returned empty results")
                else:
                    raise ValueError("Could not load emotion classifier")
                
            except Exception as e:
                # If emotion analysis fails, still show the keyword-based results
                st.subheader("📝 Mental Health Classification (मानसिक स्वास्थ्य वर्गीकरण)")
                st.write(f"**🔹 Detected Issue (पहचानी गई समस्या):** {issue_label}")

                if help_resources:
                    st.subheader("📌 Suggested Resources (सुझाए गए संसाधन):")
                    for contact, link in help_resources:
                        st.write(f"📞 **{contact}**")
                        st.markdown(f"🔗 [सहायता लिंक (Help Link)]({link})")
                
                st.warning("Couldn't analyze emotions from the text.")
                # Uncomment for debugging
                # st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text for analysis. (कृपया विश्लेषण के लिए कुछ पाठ दर्ज करें।)")

st.markdown("---")
st.write("💡 *This chatbot is for informational purposes only. If you are in crisis, please seek professional help.👨🏻‍⚕️(यह चैटबॉट केवल सूचना के उद्देश्य से है। यदि आप संकट में हैं, तो कृपया पेशेवर मदद लें।)*")
