🎬 Netflix Recommendation System

  Get Personalized Movie & TV Show Suggestions Instantly!






🚀 Live Demo

  Try the app live: Streamlit Demo

  (https://movie-suggestion-project-rk9z9h6u3dhydfuie2pzx7.streamlit.app/)

📖 Project Overview

  This project is a content-based recommendation system inspired by Netflix.
  It lets users input a movie or TV show and get a list of similar content suggestions instantly.

 The system uses Python, Pandas, Scikit-learn, and a Streamlit interface to deliver fast, interactive recommendations based on a curated dataset of Netflix titles.

✨ Features

  🔍 Search any movie or TV show by name

  🎯 Content-based recommendations for similar titles

🖥️ Interactive Netflix-style UI using Streamlit
     made UI of streamlit for revommendation system with the help of AI (chatgpt)
⚡ Fast recommendations with preprocessed dataset



🗂️ File Structure
 Netflix-Recommendation-System/
 │
 ├── data_processing_for_netflix_recommendation.py   # Backend: Data cleaning & recommendation logic
 ├── NetflixUI.py                                   # Frontend: Streamlit interface
 ├── netflix_titles-2.csv.xlsx                      # Dataset: Netflix movie/TV show info
 ├── requirements.txt                               # Python dependencies
 ├── .gitignore                                     # Files to ignore in Git
 └── README.md                                      # Project documentation

💻 Tech Stack

 Python – Core programming language

 Pandas & NumPy – Data manipulation

 Scikit-learn – Content similarity & recommendation logic

 Streamlit – Interactive frontend

 Matplotlib & Seaborn – Optional visualizations

🖼️ Screenshots / Demo

  <img width="1797" height="898" alt="Screenshot 2026-01-19 185439" src="https://github.com/user-attachments/assets/85d1643d-f72b-46b5-8f30-614049442dea" />




⚡ Installation & Usage

  Clone the repository:

 git clone <your-repo-link>
 cd netflix-recommendation-system


Create a virtual environment:

 python -m venv venv


Activate the environment:

 Windows: venv\Scripts\activate

 Mac/Linux: source venv/bin/activate

 Install dependencies:

 pip install -r requirements.txt

 Run the Streamlit app:

 streamlit run NetflixUI.py


Enter a movie/TV show name → enjoy recommendations!

📝 Dataset

 File: netflix_titles-2.csv.xlsx

 Contains: title, type, description, genre, release_year, rating

⚠️ Make sure the dataset is in the same directory as your project.

🔮 Future Improvements

 Add collaborative filtering for hybrid recommendations

 Incorporate ratings/reviews analysis

 Use advanced NLP models for better description similarity

 Enhance UI with richer Netflix-like design

🛠️ How It Works
 
 Load the Netflix dataset

 Clean and preprocess title, description, and genre

 Build a content-based similarity model

 Streamlit UI takes user input

 Returns recommended similar content instantly

📄 License

 This project is licensed under the MIT License.
