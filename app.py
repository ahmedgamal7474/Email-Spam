import streamlit as st
import pickle
import numpy as np


model = pickle.load(open('E:\Project\email_class.pkl', 'rb'))
vectorizer = pickle.load(open('E:\Project\count_vect', 'rb'))

st.title("📧 Email Spam Classifier")
st.write("Classify emails as Spam or Not Spam using a Naive Bayes Classifier.")


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

