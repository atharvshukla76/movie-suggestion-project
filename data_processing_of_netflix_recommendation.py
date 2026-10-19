import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "netflix_titles-2.csv.xlsx")
CSV_PATH = os.path.join(BASE_DIR, "netflix_clean.csv")
CACHE_PATH = os.path.join(BASE_DIR, "model_cache.pkl")

features = [
    'duration_minutes',
    'is_season',
    'release_year',
    'rating_enc',
    'genre_count',
    'country_enc'
]


def _build_and_save_cache():
    """Reads Excel, trains models, and saves fast cache files to disk."""
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Dataset not found: {EXCEL_PATH}")

    print("⏳ Building fast model cache for the first time... (Takes ~15s)")
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ['duration', 'listed_in', 'release_year', 'rating', 'country', 'type', 'title']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Feature Engineering
    df['duration'] = df['duration'].astype(str)
    df['duration_minutes'] = df['duration'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['is_season'] = df['duration'].str.contains('season', case=False, na=False).astype(int)
    df['genre_count'] = df['listed_in'].astype(str).str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)

    rating_le = LabelEncoder()
    country_le = LabelEncoder()
    type_le = LabelEncoder()

    df['rating_enc'] = rating_le.fit_transform(df['rating'].fillna('Unknown').astype(str))
    df['country_enc'] = country_le.fit_transform(df['country'].fillna('Unknown').astype(str))
    y_encoded = type_le.fit_transform(df['type'].astype(str))
    type_mapping = dict(zip(type_le.transform(type_le.classes_), type_le.classes_))

    # Train Random Forest Classifier
    X = df[features].fillna(0)
    y = y_encoded
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    target_names = [str(c) for c in type_le.classes_]
    report_text = classification_report(y_test, y_pred, target_names=target_names)

    # Content-Based TF-IDF Similarity
    df['content_features'] = (
        df['listed_in'].fillna('').astype(str) + ' ' +
        (df['description'].fillna('').astype(str) if 'description' in df.columns else '') + ' ' +
        df['country'].fillna('').astype(str) + ' ' +
        df['rating'].fillna('').astype(str)
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index, index=df['title'].astype(str).str.strip().str.lower())
    indices = indices[~indices.index.duplicated(keep='first')]

    # Save CSV and Pickle Cache
    df.to_csv(CSV_PATH, index=False)
    cache_data = {
        "model": model,
        "type_mapping": type_mapping,
        "cosine_sim": cosine_sim,
        "indices": indices,
        "classification_report": report_text
    }
    joblib.dump(cache_data, CACHE_PATH)
    print("⚡ Fast cache saved successfully! Future startups will load in 0.1s.")

    return df, model, type_mapping, cosine_sim, indices, report_text


# Load from fast cache if available, otherwise build once
if os.path.exists(CACHE_PATH) and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    cache = joblib.load(CACHE_PATH)
    model = cache["model"]
    type_mapping = cache["type_mapping"]
    cosine_sim = cache["cosine_sim"]
    indices = cache["indices"]
    classification_report_text = cache["classification_report"]
else:
    df, model, type_mapping, cosine_sim, indices, classification_report_text = _build_and_save_cache()


import difflib

def predict_for_title(user_title, top_n=5):
    """
    Given a title (from UI), return predictions and similar recommendations.
    Supports exact, partial, and fuzzy typo matching (e.g., 'stanger things').
    """
    clean_user_title = user_title.strip().lower()

    # 1. Direct or partial substring match
    matches = df[df['title'].astype(str).str.lower().str.contains(clean_user_title, regex=False)]
    
    # 2. If no direct match, attempt Fuzzy Typo Matching (e.g. 'stanger things' -> 'Stranger Things')
    if matches.empty:
        all_titles = df['title'].dropna().astype(str).tolist()
        close_titles = difflib.get_close_matches(clean_user_title, all_titles, n=3, cutoff=0.6)
        if close_titles:
            matches = df[df['title'].isin(close_titles)]

    predictions = []
    if not matches.empty:
        user_X = matches[features].fillna(0)
        preds = model.predict(user_X)
        labels = [type_mapping[p] for p in preds]
        predictions = list(zip(matches['title'], labels))

    recommendations = []
    target_title = None

    if clean_user_title in indices:
        target_title = clean_user_title
    elif not matches.empty:
        target_title = str(matches.iloc[0]['title']).strip().lower()

    if target_title and target_title in indices:
        idx = indices[target_title]
        if hasattr(idx, '__iter__'):
            idx = idx[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommendations = [(df.iloc[i[0]]['title'], df.iloc[i[0]]['type']) for i in sim_scores]

    return predictions, recommendations



def get_classification_report():
    """Return classification report string"""
    return classification_report_text


def plot_data_distributions():
    """Generate exploratory data analysis plots. Called only when run standalone."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fig, axs = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)), dpi=95)
        if len(numeric_cols) == 1:
            axs = [axs]

        for ax, col in zip(axs, numeric_cols):
            ax.hist(df[col], bins=30)
            ax.set_title(f'Histogram of {col}')
            ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()

    corr = df[numeric_cols].corr()
    plt.figure(dpi=130)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    plt.pie(
        df['type'].value_counts(),
        labels=df['type'].value_counts().index,
        autopct='%1.1f%%',
        shadow=True
    )
    plt.title('Distribution of Netflix Content')
    plt.show()


if __name__ == '__main__':
    plot_data_distributions()
    print("Classification Report:\n", get_classification_report())
