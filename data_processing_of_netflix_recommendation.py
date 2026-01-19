import os
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

# ======================
# Load data safely
# ======================
file_path = r"C:\Users\Atharv Shukla\OneDrive\Desktop\my netflix project\netflix_titles-2.csv.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_excel(file_path)

# Normalize column names (PREVENTS 90% bugs)
df.columns = df.columns.str.strip().str.lower()

# ======================
# Column safety checks
# ======================
required_cols = ['duration', 'listed_in', 'release_year', 'rating', 'country', 'type', 'title']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# ======================
# Feature Engineering
# ======================
df['duration'] = df['duration'].astype(str)

df['duration_minutes'] = (
    df['duration']
    .str.extract(r'(\d+)')
    .fillna(0)
    .astype(int)
)

df['is_season'] = df['duration'].str.contains('season', case=False, na=False).astype(int)

df['genre_count'] = (
    df['listed_in']
    .astype(str)
    .str.split(',')
    .apply(lambda x: len(x) if isinstance(x, list) else 0)
)

df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)

# ======================
# Encode categorical columns
# ======================
rating_le = LabelEncoder()
country_le = LabelEncoder()
type_le = LabelEncoder()

df['rating_enc'] = rating_le.fit_transform(df['rating'].fillna('Unknown').astype(str))
df['country_enc'] = country_le.fit_transform(df['country'].fillna('Unknown').astype(str))
y_encoded = type_le.fit_transform(df['type'].astype(str))

# ======================
# EDA
# ======================
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

# ======================
# Model Training
# ======================
features = [
    'duration_minutes',
    'is_season',
    'release_year',
    'rating_enc',
    'genre_count',
    'country_enc'
]

X = df[features].fillna(0)
y = y_encoded

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ======================
# Classification Report (Bulletproof)
# ======================
target_names = [str(c) for c in type_le.classes_]

# Return string instead of printing
classification_report_text = classification_report(y_test, y_pred, target_names=target_names)

# ======================
# CONTENT-BASED RECOMMENDATION (ADDED FEATURE)
# ======================
# Combine text features safely
df['content_features'] = (
    df['listed_in'].fillna('').astype(str) + ' ' +
    (df['description'].fillna('').astype(str) if 'description' in df.columns else '') + ' ' +
    df['country'].fillna('').astype(str) + ' ' +
    df['rating'].fillna('').astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content_features'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(df.index, index=df['title'].astype(str).str.lower()).drop_duplicates()

# ======================
# Streamlit-safe user functions
# ======================
def predict_for_title(user_title, top_n=5):
    """
    Given a title (from UI), return predictions and similar recommendations.
    All original logic retained, no input()/print() anywhere.
    """
    user_title = user_title.strip().lower()

    # Prediction
    matches = df[df['title'].astype(str).str.lower().str.contains(user_title, regex=False)]
    predictions = []
    if not matches.empty:
        user_X = matches[features].fillna(0)
        preds = model.predict(user_X)
        labels = type_le.inverse_transform(preds)
        predictions = list(zip(matches['title'], labels))

    # Recommendations
    recommendations = []
    if user_title in indices:
        idx = indices[user_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommendations = [(df.iloc[i[0]]['title'], df.iloc[i[0]]['type']) for i in sim_scores]

    return predictions, recommendations

def get_classification_report():
    """Return classification report string"""
    return classification_report_text
