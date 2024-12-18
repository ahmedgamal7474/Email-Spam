import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import requests
from streamlit_lottie import st_lottie  

# Load the pre-trained model and vectorizer
model = pickle.load(open('E:/Project/email_class.pkl', 'rb'))
vectorizer = pickle.load(open('E:/Project/count_vect', 'rb'))

# Load dataset for visualization
df = pd.read_csv('Spam Email Classifier.csv')

# Streamlit App Configuration
st.set_page_config(page_title="Email Spam Classifier", layout="wide")

def load_lottie_url(url):
    """Load Lottie animation from a URL"""
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Add your animation URL (replace with a Lottie URL you like)
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_animation = load_lottie_url(lottie_url)

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "Email Classification", "Data Visualization", "About"])

# Welcome Page
if page == "Home":
    st.title("📧 Email Spam Classifier")
    if lottie_animation:
        st_lottie(
            lottie_animation,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height=300,
            width=600,
        )

    st.write("""
        This project uses a **Naive Bayes Classifier** to classify emails as Spam or Not Spam. 
        The model is trained on a dataset of labeled emails, and it uses text analysis techniques 
        such as **Count Vectorization** to transform the text data into numerical features for classification.
    """)
    st.write("""
        ### Features:
        - **Classify Your Email**: Enter email content to check if it is spam or not.
        - **Visualizations**: Explore insights about the dataset with word clouds and distribution plots.
        """)

# Email Classification Page
# Email Classification Page
elif page == "Email Classification":
    st.title("📧 Email Spam Classifier")
    st.write("Classify emails as Spam or Not Spam using a Naive Bayes Classifier.")

    # Animation URLs
    lottie_url_classify_1 = "https://lottie.host/1dd99b5a-1328-48fc-968a-2e62c3ec8b45/FezRifEZOx.json"
    lottie_url_classify_2 = "https://lottie.host/fb438f57-ba71-4ace-af35-42c69151f3f9/t8yarEFgS1.json"  

    # Load animations
    lottie_animation_classify_1 = load_lottie_url(lottie_url_classify_1)
    lottie_animation_classify_2 = load_lottie_url(lottie_url_classify_2)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        if lottie_animation_classify_1:
            st_lottie(
                lottie_animation_classify_1,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=600,
            )
        else:
            st.warning("Lottie animation 1 could not be loaded.")

    with col2:
        if lottie_animation_classify_2:
            st_lottie(
                lottie_animation_classify_2,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=300,
                width=600,
            )
        else:
            st.warning("Lottie animation 2 could not be loaded.")

    st.subheader("Classify Your Email")
    email_text = st.text_area("Enter the email content here:")

    if st.button("Classify Email"):
        if email_text:
            email_count = vectorizer.transform([email_text])
            prediction = model.predict(email_count)

            if prediction[0] == 1:
                st.error("🚨 This email is classified as SPAM!")
            else:
                st.success("✅ This email is classified as NOT SPAM!")
        else:
            st.warning("Please enter some text to classify.")

# Visualizations Page
elif page == "Data Visualization":
    st.title("📊 Data Visualizations")
    st.write("Explore insights from the dataset used for training the model.")

    # Word Cloud for Spam
    st.subheader("Word Cloud: Spam Emails")
    spam_words = ' '.join(list(df[df['label'] == 1]['text']))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(spam_wc, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    # Word Cloud for Ham
    st.subheader("Word Cloud: Non-Spam Emails")
    ham_words = ' '.join(list(df[df['label'] == 0]['text']))
    ham_wc = WordCloud(width=512, height=512).generate(ham_words)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(ham_wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    # Spam vs Ham Count Plot
    st.subheader("Spam vs Ham Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='label', data=df, ax=ax3)
    ax3.set_title('Spam vs Ham Distribution')
    st.pyplot(fig3)

elif page == "About":
    st.title("👨‍💻 About")
    st.write("### Members of the Developers Team")
    
    # Team Members
    st.write("""
        - Ahmed Gamal Ahmed  
        - Adham Mohamed Abu Zeid  
        - Ibrahim Gamal Abdel Nasser  
        - Momen Bakr Al-Siddiq  
    """)

    st.write("---")  # Separator line
    
    # Contact Information
    st.write("### Contact Information")
    
    # WhatsApp
    st.write("📱 **WhatsApp**")
    st.markdown("""
        <a href="https://wa.me/01064937645" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="30"/>
            01064937645
        </a>
        """, unsafe_allow_html=True)
    
    # LinkedIn
    st.write("🔗 **LinkedIn**")
    st.markdown("""
        <a href="https://www.linkedin.com/in/ahmedgamal7474?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" width="30"/>
            Team's LinkedIn
        </a>
        """, unsafe_allow_html=True)
    
    # GitHub
    st.write("💻 **GitHub**")
    st.markdown("""
        <a href="https://github.com/ahmedgamal7474" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="30"/>
            Team's GitHub
        </a>
        """, unsafe_allow_html=True)