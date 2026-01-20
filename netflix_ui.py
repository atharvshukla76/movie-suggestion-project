import streamlit as st
from data_processing_of_netflix_recommendation import (
    predict_for_title,
    get_classification_report
)


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
)


if "search_history" not in st.session_state:
    st.session_state.search_history = {}


st.markdown(
    """
    <div style='background-color:#e50914;padding:12px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>Netflix Recommendation System 🎬</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### Smart recommendations based on your input & interest")


def human_type(value):
    value = str(value).lower()
    if value in ["0", "movie"]:
        return "Movie"
    if value in ["1", "tv show", "tvshow"]:
        return "TV Show"
    return value


user_input = st.text_input(
    "Search for a Movie or TV Show",
    placeholder="Start typing... (min 3 characters)"
)


if len(user_input.strip()) >= 3:
    preds, recs = predict_for_title(user_input)

    
    st.session_state.search_history[user_input] = (
        st.session_state.search_history.get(user_input, 0) + 1
    )

    
    st.markdown("#### 🔍 Prediction")
    if preds:
        for title, label in preds:
            st.success(f"{title} → {human_type(label)}")
    else:
        st.warning("No matching titles found")

    
    st.markdown("#### 🎯 Recommended for You")
    if recs:
        for title, type_ in recs:
            st.info(f"{title} ({human_type(type_)})")
    else:
        st.warning("No recommendations available")


if st.session_state.search_history:
    st.markdown("#### 📊 Your Interest Pattern")
    for k, v in sorted(
        st.session_state.search_history.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        st.write(f"• **{k}** → searched {v} times")


with st.expander("📄 Model Classification Report"):
    st.text(get_classification_report())


st.markdown(
    """
    <hr style='border:1px solid #e50914'>
    <p style='text-align:center;color:gray;font-size:12px'>
    Streamlit-based Recommendation System (Academic / Portfolio level)
    </p>
    """,
    unsafe_allow_html=True
)
