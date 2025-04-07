import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'  # Set writable cache directory

import pandas as pd
import streamlit as st
import google.generativeai as genai
import json
from google.api_core import exceptions
import requests
from bs4 import BeautifulSoup
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
    embedding_model = SentenceTransformer('./model')  # Load pre-downloaded model
except ImportError as e:
    st.error(f"Warning: Failed to import sentence_transformers or faiss due to: {e}. Running without embeddings. Install compatible versions or pre-download the model.")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    faiss = None
    np = None
    embedding_model = None

# Configure Gemini API
API_KEY = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=API_KEY)

# Load the dataset
try:
    df = pd.read_csv("shl_assessments_updated.csv")
    df = df[df['Test Type'].str.contains("Knowledge & Skills|Ability & Aptitude|Assessment Exercises", case=False, na=False)]
except FileNotFoundError:
    st.error("Error: 'shl_assessments_updated.csv' not found in the directory. Please add the file.")
    df = pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support", "Duration", "Test Type"])
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    df = pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support", "Duration", "Test Type"])

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = " ".join(p.text.strip() for p in soup.find_all('p') if p.text.strip())
        return text if text else "No description available"
    except requests.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return "No description available"

# Function to create embeddings and FAISS index (if available)
def setup_vector_database(df):
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        return None, [], []
    descriptions = [f"{row['Assessment Name']} {extract_text_from_url(row['URL'])}" for _, row in df.iterrows()]
    embeddings = embedding_model.encode(descriptions, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) if faiss else None
    if index:
        index.add(embeddings)
    return index, descriptions, embeddings

# Setup FAISS index
index, descriptions, embeddings = setup_vector_database(df)

# Function to retrieve similar assessments
def retrieve_similar_assessments(query, k=10):
    if not EMBEDDINGS_AVAILABLE or index is None or embedding_model is None:
        return [(row, 0) for _, row in df.head(k).iterrows()]
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [(df.iloc[i], distances[0][j]) for j, i in enumerate(indices[0]) if i < len(df)]

# Function to parse query with Gemini
def parse_query_with_gemini(query):
    try:
        similar_assessments = retrieve_similar_assessments(query, k=5)
        context = "\n".join([f"Assessment: {assess['Assessment Name']}, Type: {assess['Test Type']}" for assess, _ in similar_assessments])

        prompt = f"""
        Based on the following query or job description and the provided context, extract the required skills, maximum assessment duration in minutes,
        and select relevant test types from this list: ['Ability & Aptitude', 'Assessment Exercises', 'Biodata & Situational Judgement',
        'Competencies', 'Development & 360', 'Knowledge & Skills', 'Personality & Behavior', 'Simulations'].
        Return the result in JSON format. If not specified, use default values (skills: [], max_duration: None, test_types: []).
        Query: {query}
        Context: {context}
        """
        response = genai.generate_text(model="gemini-pro", prompt=prompt, temperature=0.2)
        return json.loads(response.text.strip()) if response.text.strip().startswith("{") else {"skills": [], "max_duration": None, "test_types": []}
    except exceptions.GoogleAPIError as e:
        st.error(f"Gemini API error: {e}")
        return {"skills": [], "max_duration": None, "test_types": []}
    except json.JSONDecodeError:
        st.error("Failed to parse Gemini response as JSON")
        return {"skills": [], "max_duration": None, "test_types": []}

# Function to recommend assessments
def recommend_assessments(query, max_results=10):
    requirements = parse_query_with_gemini(query)
    skills = [s.lower().strip() for s in requirements.get("skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]

    similar_assessments = retrieve_similar_assessments(query, k=max_results)
    recommendations = []

    for (row, similarity_distance), _ in zip(similar_assessments, range(max_results)):
        score = 0
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]

        skill_matches = sum(1 for skill in skills if skill in assessment_name)
        score += skill_matches * 50

        if max_duration and duration != "N/A" and float(duration) <= float(max_duration):
            score += 20

        if required_test_types and any(test_type in ", ".join(test_types) for test_type in required_test_types):
            score += 30

        similarity_score = 100 - (similarity_distance / np.max(similarity_distance) * 50) if EMBEDDINGS_AVAILABLE and similarity_distance > 0 and np is not None else 0
        score += similarity_score

        if score > 0:
            recommendations.append((score, row))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations[:max_results] if recommendations else []

# Streamlit app
def run_streamlit():
    st.title("SHL Assessment Recommendation System")
    st.write("Enter a job description text or URL to get the top 10 relevant Individual Test Solutions.")

    user_input = st.text_area("Enter Job Description Text or URL", height=200)

    if st.button("Get Recommendations"):
        if user_input:
            if user_input.startswith(('http://', 'https://')):
                user_input = extract_text_from_url(user_input)

            st.write("Parsed Requirements:", parse_query_with_gemini(user_input))
            results = recommend_assessments(user_input, max_results=10)

            if results:
                st.write("Top 10 Recommendations:")
                for _, row in results:
                    st.write(f"{row['Assessment Name']} - [Link]({row['URL']}) - Remote: {row['Remote Testing Support']} - Adaptive: {row['Adaptive/IRT Support']} - Duration: {row['Duration']} - Type: {row['Test Type']}")
            else:
                st.write("No matching assessments found.")
        else:
            st.write("Please enter a job description text or URL.")

    st.write("Debug Info:", parse_query_with_gemini(user_input) if user_input else "No input yet.")

if __name__ == '__main__':
    run_streamlit()