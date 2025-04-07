import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import pandas as pd
import streamlit as st
import google.generativeai as genai
import json
import requests
from bs4 import BeautifulSoup
from google.api_core import exceptions
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide", initial_sidebar_state="expanded")

# Cache model initialization with retry logic
@st.cache_resource
def load_model(max_retries=3, delay=5):
    from sentence_transformers import SentenceTransformer
    for attempt in range(max_retries):
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Confirmed model path
            st.success("Successfully loaded SentenceTransformer model.")
            return model
        except (ImportError, OSError, ValueError) as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed to load model due to: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.warning(f"Failed to load SentenceTransformer model after {max_retries} attempts due to: {e}. Running without embeddings.")
                return None

# Initialize embedding model and dependencies
embedding_model = load_model()
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = embedding_model is not None
except ImportError as e:
    st.warning(f"Failed to import sentence_transformers or faiss due to: {e}. Running without embeddings.")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    faiss = None

# Configure Gemini API
API_KEY = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=API_KEY)

# Load and cache dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("shl_assessments_updated.csv")
        return df[df['Test Type'].str.contains("Knowledge & Skills|Ability & Aptitude|Assessment Exercises", case=False, na=False)]
    except FileNotFoundError:
        st.error("Error: 'shl_assessments_updated.csv' not found. Using empty dataset.")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support", "Duration", "Test Type"])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support", "Duration", "Test Type"])

df = load_dataset()

# Cache vector database setup
@st.cache_data
def setup_vector_database(_df):
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        return None, [], []
    descriptions = [f"{row['Assessment Name']} {row.get('URL', '')} {' '.join(row['Assessment Name'].lower().split())}" for _, row in _df.iterrows()]
    embeddings = embedding_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) if faiss else None
    if index:
        index.add(embeddings)
    return index, descriptions, embeddings

index, descriptions, embeddings = setup_vector_database(df)

# Advanced URL extraction with threading
def extract_text_from_url_threaded(url, result_queue):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        description_divs = soup.find_all('div', class_=['description', 'job-details', 'show-more-less-html', 'description__text'])
        if not description_divs:
            description_divs = soup.find_all('section', class_=['description'])

        text = ""
        for div in description_divs:
            text += " ".join(p.text.strip() for p in div.find_all('p') if p.text.strip())
            text += " ".join(span.text.strip() for span in div.find_all('span') if span.text.strip())
            text += " ".join(li.text.strip() for li in div.find_all('li') if li.text.strip())
            text += " ".join(script.text.strip() for script in div.find_all('script') if 'description' in script.text.lower())

        if not text:
            text = " ".join(soup.body.get_text(separator=" ").strip().split())

        result_queue.put(text if text else "No description available")
    except requests.RequestException as e:
        result_queue.put(f"Error fetching URL {url}: {e}")

@st.cache_data
def extract_text_from_url(url):
    result_queue = Queue()
    thread = threading.Thread(target=extract_text_from_url_threaded, args=(url, result_queue))
    thread.start()
    thread.join(timeout=15)
    return result_queue.get() if not thread.is_alive() else "Timeout or error fetching URL"

# Cache similar assessments with threading
def retrieve_similar_assessments_threaded(query, k, result_queue, max_retries=3):
    if not EMBEDDINGS_AVAILABLE or index is None or embedding_model is None:
        result_queue.put([(row, 0) for _, row in df.head(k).iterrows()])
        return
    for attempt in range(max_retries):
        try:
            query_embedding = embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            distances, indices = index.search(query_embedding, k)
            result_queue.put([(df.iloc[i], distances[0][j]) for j, i in enumerate(indices[0]) if i < len(df)])
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            result_queue.put([(row, 0) for _, row in df.head(k).iterrows()])
            st.error(f"Failed to retrieve similar assessments after {max_retries} attempts: {e}")

@st.cache_data
def retrieve_similar_assessments(query, k=10):
    result_queue = Queue()
    thread = threading.Thread(target=retrieve_similar_assessments_threaded, args=(query, k, result_queue))
    thread.start()
    thread.join(timeout=10)
    return result_queue.get() if not thread.is_alive() else [(row, 0) for _, row in df.head(k).iterrows()]

# Enhanced Gemini parsing with threading
def parse_query_with_gemini_threaded(query, result_queue, max_retries=3):
    def get_response(prompt):
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }
        client = genai.GenerativeModel(model_name="models/gemini-2.0-flash", generation_config=generation_config)
        return client.generate_content(prompt)

    try:
        if query.startswith(('http://', 'https://')):
            query = extract_text_from_url(query)

        similar_assessments = retrieve_similar_assessments(query, k=5)
        context = "\n".join([f"Assessment: {assess['Assessment Name']}, Type: {assess['Test Type']}, Duration: {assess['Duration']}" for assess, _ in similar_assessments])

        prompt = f"""
        You are an expert in job assessment analysis. Analyze the following query (job description or URL-extracted text) and context to extract:
        - Required skills (e.g., Java, Python, SQL, .NET Framework, or any technical skills implied, listed explicitly or inferred).
        - Maximum assessment duration in minutes (extracted directly or inferred from job timeline, e.g., 'complete in 1 hour' → 60, default to null if unclear).
        - Relevant test types from: ['Ability & Aptitude', 'Assessment Exercises', 'Biodata & Situational Judgement', 'Competencies', 'Development & 360', 'Knowledge & Skills', 'Personality & Behavior', 'Simulations'] (inferred based on skills and context).
        Return a JSON object with keys 'required_skills' (list), 'max_duration' (number or null), and 'test_types' (list). Use defaults if data is missing: required_skills: [], max_duration: null, test_types: [].
        Query: {query}
        Context: {context}
        """

        for attempt in range(max_retries):
            try:
                response = get_response(prompt)
                result = json.loads(response.text) if response.text and response.text.strip().startswith("{") else {"required_skills": [], "max_duration": None, "test_types": []}
                if not result.get("required_skills"):
                    result["required_skills"] = [skill for skill in ["java", "python", "sql", ".net framework"] if skill in query.lower()]
                if not result.get("test_types"):
                    result["test_types"] = ["Knowledge & Skills"] if any(skill in query.lower() for skill in ["java", "python", "sql", ".net"]) else []
                if not result.get("max_duration") and "minutes" in query.lower():
                    import re
                    match = re.search(r"(\d+)\s*minutes?", query.lower())
                    result["max_duration"] = int(match.group(1)) if match else None
                result_queue.put(result)
                return
            except exceptions.GoogleAPIError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                result_queue.put({"required_skills": [], "max_duration": None, "test_types": []})
                st.error(f"Gemini API error after {max_retries} attempts: {e}")
            except (json.JSONDecodeError, AttributeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                result_queue.put({"required_skills": [], "max_duration": None, "test_types": []})
                st.error(f"Parsing error after {max_retries} attempts: {e}")

    except Exception as e:
        result_queue.put({"required_skills": [], "max_duration": None, "test_types": []})
        st.error(f"Unexpected error in parse_query_with_gemini: {e}")

@st.cache_data
def parse_query_with_gemini(query):
    result_queue = Queue()
    thread = threading.Thread(target=parse_query_with_gemini_threaded, args=(query, result_queue))
    thread.start()
    thread.join(timeout=15)
    return result_queue.get() if not thread.is_alive() else {"required_skills": [], "max_duration": None, "test_types": []}

# Optimized recommendations with detailed scoring
@st.cache_data
def recommend_assessments(query, max_results=10):
    requirements = parse_query_with_gemini(query)
    required_skills = [s.lower().strip() for s in requirements.get("required_skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]

    similar_assessments = retrieve_similar_assessments(query, k=max_results * 2)
    recommendations = []

    for (row, similarity_distance), _ in zip(similar_assessments, range(max_results * 2)):
        score = 0
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]

        skill_matches = sum(1 for skill in required_skills if skill in assessment_name or any(skill in t.lower() for t in test_types))
        if skill_matches == len(required_skills):
            score += 100
        elif skill_matches > 0:
            score += skill_matches * 40

        if max_duration and duration != "N/A" and float(duration) <= float(max_duration) * 1.2:
            score += 30
        elif not max_duration and duration != "N/A" and float(duration) <= 60:
            score += 15

        if required_test_types and any(test_type in ", ".join(test_types) for test_type in required_test_types):
            score += 50
        elif not required_test_types and any(t in ["Knowledge & Skills", "Ability & Aptitude"] for t in test_types):
            score += 20

        similarity_score = 100 - (similarity_distance / np.max(similarity_distance) * 50) if EMBEDDINGS_AVAILABLE and similarity_distance > 0 and np is not None else 0
        score += similarity_score * 0.3

        if score > 0:
            recommendations.append((score, row))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations[:max_results] if recommendations else []

# Simulate evaluation metrics with corrected logic
def evaluate_recommendations():
    test_queries = [
        {"query": "Hiring Java, Python, SQL developers with .NET Framework, 40 minutes", "relevant": ["Java Programming", "Python Programming", "SQL Server", ".NET Framework 4.5"]},
        {"query": "Research Engineer AI, 60 minutes", "relevant": ["AI Fundamentals", "Research Skills"]},
    ]
    k = 5
    recall_scores = []
    ap_scores = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(recommend_assessments, test["query"], k) for test in test_queries]
        results = [future.result() for future in futures]

    for i, test in enumerate(test_queries):
        recommended_names = [result[1]["Assessment Name"] for result in results[i]]
        relevant_set = set(test["relevant"])
        retrieved_relevant = len(set(recommended_names) & relevant_set)
        total_relevant = len(relevant_set)

        # Recall@K
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
        recall_scores.append(recall)

        # Average Precision@K
        precision_at_k = 0
        relevant_count = 0
        for j, name in enumerate(recommended_names[:k], 1):
            if name in relevant_set:
                relevant_count += 1
                precision_at_k += relevant_count / j
        ap = precision_at_k / min(k, total_relevant) if min(k, total_relevant) > 0 else 0
        ap_scores.append(ap)

    mean_recall = np.mean(recall_scores) if recall_scores else 0
    map_k = np.mean(ap_scores) if ap_scores else 0
    return mean_recall, map_k

# Improved UI with tabular format and clickable links
def run_streamlit():
    st.title("SHL Assessment Recommendation System :rocket:")
    st.markdown("**Welcome!** Enter a job description or URL (e.g., LinkedIn job page) to get tailored assessment recommendations. :chart_with_upwards_trend:")

    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Input Type", ["Text", "URL"], index=1)
    max_results = st.sidebar.slider("Max Recommendations", 5, 15, 10)

    user_input = st.text_area(
        "Enter Job Description or URL",
        height=150,
        placeholder="E.g., 'Hiring Java, Python, SQL developers with .NET Framework, 40 minutes' or a URL",
        value="https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in" if input_type == "URL" else "I am hiring for Java, Python, and SQL developers with .NET Framework experience, 40 minutes."
    )

    if st.button("Generate Recommendations :mag_right:"):
        if user_input:
            with st.spinner("Analyzing input and generating recommendations..."):
                st.subheader("Parsed Requirements")
                requirements = parse_query_with_gemini(user_input)
                st.json(requirements)

                st.subheader("Top Recommendations")
                results = recommend_assessments(user_input, max_results=max_results)
                if results:
                    # Create table data with clickable links
                    table_data = []
                    for i, (_, row) in enumerate(results):
                        table_data.append({
                            "Rank": i + 1,
                            "Assessment Name": row["Assessment Name"],
                            "URL": row["URL"],  # Store raw URL for rendering
                            "Duration (min)": float(row["Duration"]) if row["Duration"] != "N/A" else "N/A",
                            "Remote Testing Support": row["Remote Testing Support"],
                            "Adaptive/IRT Support": row["Adaptive/IRT Support"],
                            "Test Type": row["Test Type"]
                        })
                    recommendations_df = pd.DataFrame(table_data)

                    # Render table with HTML for clickable links
                    html_table = recommendations_df.to_html(escape=False, index=False)
                    html_table = html_table.replace('<td>', '<td style="text-align: left; padding: 10px;">')
                    html_table = html_table.replace('<th>', '<th style="background-color: #333333; color: #ffffff; font-weight: bold; padding: 10px; border-bottom: 2px solid #555;">')
                    html_table = html_table.replace('<tr>', '<tr style="background-color: #1a1a1a; color: #ffffff;">')
                    html_table = html_table.replace('</tr>', '</tr><tr style="background-color: #1a1a1a; color: #ffffff;">')  # Ensure consistent row styling
                    html_table = html_table.replace('<table border="1" class="dataframe">', '<table border="1" class="dataframe" style="background-color: #1a1a1a; color: #ffffff; width: 100%;">')
                    # Replace URL column with clickable links
                    html_table = html_table.replace('<td>' + recommendations_df["URL"][0] + '</td>',
                                                   '<td><a href="' + recommendations_df["URL"][0] + '" target="_blank" style="color: #1E90FF; text-decoration: underline;">Link</a></td>')
                    for i in range(1, len(recommendations_df)):
                        html_table = html_table.replace('<td>' + recommendations_df["URL"][i] + '</td>',
                                                       '<td><a href="' + recommendations_df["URL"][i] + '" target="_blank" style="color: #1E90FF; text-decoration: underline;">Link</a></td>')

                    # Add hover effect
                    html_table = html_table.replace('</tr>', '</tr><tr style="background-color: #2a2a2a;" onmouseover="this.style.backgroundColor=\'#2a2a2a\';" onmouseout="this.style.backgroundColor=\'#1a1a1a\';">')

                    st.write(html_table, unsafe_allow_html=True)

                else:
                    st.error("No matching assessments found. Please check the input or dataset.")

                st.subheader("Evaluation Metrics")
                mean_recall, map_k = evaluate_recommendations()
                st.write(f"Mean Recall@5: {mean_recall:.3f}")
                st.write(f"Mean Average Precision @5 (MAP@5): {map_k:.3f}")

            st.subheader("Debug Info")
            st.json(parse_query_with_gemini(user_input))
        else:
            st.warning("Please enter a job description or URL.")

    st.sidebar.markdown("**Developed by xAI** :star2:")

if __name__ == '__main__':
    run_streamlit()