
import streamlit as st
from data_processing_of_netflix_recommendation import predict_for_title, get_classification_report


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <div style='background-color:#e50914;padding:10px;border-radius:10px'>
        <h1 style='color:white;text-align:center;font-family:sans-serif;'>Netflix Recommendation System 🎬</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### Find out if a title is a Movie or TV Show and get similar recommendations!")

with st.container():
    user_input = st.text_input("Enter a movie or TV show title", "")

if user_input:
    preds, recs = predict_for_title(user_input)

  
    st.markdown("#### Prediction")
    if preds:
        for title, label in preds:
            st.success(f"{title} → {label}")
    else:
        st.warning("No matching titles found.")

    
    st.markdown("#### Recommended Similar Titles")
    if recs:
        for title, type_ in recs:
            st.info(f"{title} ({type_})")
    else:
        st.warning("No recommendations found.")


with st.expander("Show Model Classification Report"):
    st.text(get_classification_report())


st.markdown(
    """
    <hr style='border:1px solid #e50914'>
    <p style='text-align:center;color:gray;font-size:12px'>
    Developed using Python, Streamlit & scikit-learn | Netflix-like Theme
    </p>
    """,
    unsafe_allow_html=True
)

