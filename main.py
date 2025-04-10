import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
import os
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
from fastapi import FastAPI, HTTPException
import uvicorn
import asyncio
import re

# Configure environment and API
API_KEY = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=API_KEY)

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")

# Initialize FastAPI app
app = FastAPI()

# Initialize Streamlit application
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner=True)
def load_model():
    """Robust model loading with download prevention and proper caching."""
    try:
        from sentence_transformers import SentenceTransformer
        import os

        model_name = "all-MiniLM-L6-v2"
        model_dir = os.path.join("models", model_name)
        os.makedirs(model_dir, exist_ok=True)

        required_files = ["config.json", "pytorch_model.bin", "sentence_bert_config.json"]
        files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

        if files_exist:
            try:
                model = SentenceTransformer(model_dir)
                st.success("Loaded model from local cache")
                return model
            except Exception as e:
                st.warning(f"Local model load failed: {e}. Attempting download...")

        if os.getenv('HF_HUB_OFFLINE', '0') == '0':
            with st.spinner("Downloading model (first time only)..."):
                try:
                    model = SentenceTransformer(f'sentence-transformers/{model_name}')
                    model.save(model_dir)
                    st.success("Model downloaded and cached")
                    return model
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    return None
        else:
            st.warning("Running in offline mode without embeddings")
            return None

    except ImportError as e:
        st.error(f"Package missing: {e}")
        return None

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

@st.cache_data
def load_dataset():
    """Load and filter the assessment dataset from output.csv."""
    try:
        df = pd.read_csv("output.csv")
        df['Duration'] = df['Duration'].fillna("N/A")
        df['Job Description'] = df['Job Description'].fillna("")
        df['Job Levels'] = df['Job Levels'].fillna("")
        df['Languages'] = df['Languages'].fillna("English (USA)")
        df['Scraped Description'] = df['Scraped Description'].fillna("")
        return df
    except FileNotFoundError:
        st.error("Error: 'output.csv' not found. Using empty dataset.")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support",
                                    "Test Type", "Duration", "Job Description", "Job Levels", "Languages", "Scraped Description"])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support",
                                    "Test Type", "Duration", "Job Description", "Job Levels", "Languages", "Scraped Description"])

df = load_dataset()

@st.cache_data
def setup_vector_database(_df):
    """Set up the FAISS vector database with embeddings using all relevant columns."""
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        return None, [], []
    descriptions = [
        f"{row['Assessment Name']} {row.get('URL', '')} {row['Job Description']} {row['Scraped Description']} "
        f"{row['Job Levels']} {row['Languages']} {' '.join(row['Assessment Name'].lower().split())}"
        for _, row in _df.iterrows()
    ]
    embeddings = embedding_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) if faiss else None
    if index:
        index.add(embeddings)
    return index, descriptions, embeddings

index, descriptions, embeddings = setup_vector_database(df)

def extract_text_from_url_threaded(url, result_queue):
    """Extract text from a URL using threading."""
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
    """Cache and extract text from a URL with timeout."""
    result_queue = Queue()
    thread = threading.Thread(target=extract_text_from_url_threaded, args=(url, result_queue))
    thread.start()
    thread.join(timeout=15)
    return result_queue.get() if not thread.is_alive() else "Timeout or error fetching URL"

def retrieve_similar_assessments_threaded(query, k, result_queue, max_retries=3):
    """Retrieve similar assessments using threading with retry logic."""
    if not EMBEDDINGS_AVAILABLE or index is None or embedding_model is None:
        result_queue.put([(row, 0) for _, row in df.iterrows()])
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
            result_queue.put([(row, 0) for _, row in df.iterrows()])
            st.error(f"Failed to retrieve similar assessments after {max_retries} attempts: {e}")

@st.cache_data
def retrieve_similar_assessments(query, k=10):
    """Cache and retrieve similar assessments."""
    result_queue = Queue()
    thread = threading.Thread(target=retrieve_similar_assessments_threaded, args=(query, k, result_queue))
    thread.start()
    thread.join(timeout=10)
    return result_queue.get() if not thread.is_alive() else [(row, 0) for _, row in df.iterrows()]

def parse_query_with_gemini_threaded(query, result_queue, max_retries=3):
    """Parse query using Gemini API with threading and retry logic."""
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
        context = "\n".join([
            f"Assessment: {assess['Assessment Name']}, Type: {assess['Test Type']}, Duration: {assess['Duration']}, "
            f"Job Description: {assess['Job Description']}, Job Levels: {assess['Job Levels']}, Languages: {assess['Languages']}"
            for assess, _ in similar_assessments
        ])
        prompt = f"""
        You are an expert in job assessment analysis. Analyze the following query and context to extract:
        - Required skills (e.g., Java, Python, SQL, .NET Framework, or any technical skills implied, listed explicitly or inferred).
        - Maximum assessment duration in minutes (extracted directly or inferred from job timeline, e.g., 'complete in 1 hour' → 60, default to null if unclear).
        - Relevant test types from: ['Ability & Aptitude', 'Assessment Exercises', 'Biodata & Situational Judgement',
          'Competencies', 'Development & 360', 'Knowledge & Skills', 'Personality & Behavior', 'Simulations']
          (inferred based on skills and context).
        - Job levels (e.g., 'Graduate', 'Mid-Professional', 'Professional Individual Contributor', inferred or explicit).
        - Preferred languages (e.g., 'English (USA)', 'English (Global)', inferred or explicit).
        Return a JSON object with keys 'required_skills' (list), 'max_duration' (number or null), 'test_types' (list),
        'job_levels' (list), 'languages' (list). Use defaults if data is missing:
        required_skills: [], max_duration: null, test_types: [], job_levels: [], languages: ['English (USA)'].
        Query: {query}
        Context: {context}
        """
        for attempt in range(max_retries):
            try:
                response = get_response(prompt)
                result = json.loads(response.text) if response.text and response.text.strip().startswith("{") else {
                    "required_skills": [], "max_duration": None, "test_types": [], "job_levels": [], "languages": ["English (USA)"]
                }
                if not result.get("required_skills"):
                    result["required_skills"] = [skill for skill in ["java", "python", "sql", ".net framework"] if skill in query.lower()]
                if not result.get("test_types"):
                    result["test_types"] = ["Knowledge & Skills"] if any(skill in query.lower() for skill in ["java", "python", "sql", ".net"]) else []
                if not result.get("max_duration") and "minutes" in query.lower():
                    match = re.search(r"(\d+)\s*minutes?", query.lower())
                    result["max_duration"] = int(match.group(1)) if match else None
                if not result.get("job_levels"):
                    result["job_levels"] = ["Mid-Professional"] if "experienced" in query.lower() else []
                if not result.get("languages"):
                    result["languages"] = ["English (USA)"]
                result_queue.put(result)
                return
            except exceptions.GoogleAPIError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                result_queue.put({"required_skills": [], "max_duration": None, "test_types": [], "job_levels": [], "languages": ["English (USA)"]})
                st.error(f"Gemini API error after {max_retries} attempts: {e}")
            except (json.JSONDecodeError, AttributeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                result_queue.put({"required_skills": [], "max_duration": None, "test_types": [], "job_levels": [], "languages": ["English (USA)"]})
                st.error(f"Parsing error after {max_retries} attempts: {e}")
    except Exception as e:
        result_queue.put({"required_skills": [], "max_duration": None, "test_types": [], "job_levels": [], "languages": ["English (USA)"]})
        st.error(f"Unexpected error in parse_query_with_gemini: {e}")

@st.cache_data
def parse_query_with_gemini(query):
    """Cache and parse query using Gemini API."""
    result_queue = Queue()
    thread = threading.Thread(target=parse_query_with_gemini_threaded, args=(query, result_queue))
    thread.start()
    thread.join(timeout=15)
    return result_queue.get() if not thread.is_alive() else {
        "required_skills": [], "max_duration": None, "test_types": [], "job_levels": [], "languages": ["English (USA)"]
    }

@st.cache_data
def recommend_assessments(query, max_results=10):
    """Generate assessment recommendations with required skills as the top priority."""
    requirements = parse_query_with_gemini(query)
    required_skills = [s.lower().strip() for s in requirements.get("required_skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]
    required_job_levels = [j.lower().strip() for j in requirements.get("job_levels", [])]
    required_languages = [l.lower().strip() for l in requirements.get("languages", [])]
    similar_assessments = retrieve_similar_assessments(query, k=max_results * 2)
    recommendations = []

    for row, similarity_distance in similar_assessments:
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]
        job_levels = [j.lower().strip() for j in str(row["Job Levels"]).split(", ")] if row["Job Levels"] else []
        languages = [l.lower().strip() for l in str(row["Languages"]).split(", ")] if row["Languages"] else []
        job_desc = row["Job Description"].lower()
        scraped_desc = row["Scraped Description"].lower()

        matched_skills = [skill for skill in required_skills if skill in assessment_name or skill in job_desc or skill in scraped_desc or any(skill in t for t in test_types)]
        skill_matches = len(matched_skills)
        skill_score = 0
        if skill_matches == len(required_skills) and required_skills:
            skill_score = 1000
        elif skill_matches > 0:
            skill_score = skill_matches * 200

        secondary_score = 0
        if max_duration and duration != "N/A" and float(duration) <= float(max_duration) * 1.2:
            secondary_score += 50
        elif not max_duration and duration != "N/A" and float(duration) <= 60:
            secondary_score += 25

        if required_test_types and any(test_type in test_types for test_type in required_test_types):
            secondary_score += 40
        elif not required_test_types and any(t in ["knowledge & skills", "ability & aptitude"] for t in test_types):
            secondary_score += 20

        if required_job_levels and any(level in job_levels for level in required_job_levels):
            secondary_score += 30
        elif not required_job_levels and "mid-professional" in job_levels:
            secondary_score += 10

        if required_languages and any(lang in languages for lang in required_languages):
            secondary_score += 20
        elif not required_languages and "english (usa)" in languages:
            secondary_score += 5

        similarity_score = 0 if not EMBEDDINGS_AVAILABLE else (100 - (similarity_distance / np.max(similarity_distance) * 50) if similarity_distance > 0 else 0)
        secondary_score += similarity_score * 0.5

        total_score = skill_score + secondary_score
        if total_score > 0:
            recommendations.append((total_score, row, matched_skills))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations[:max_results] if recommendations else []

@app.get("/recommend")
async def get_recommendations(query: str, max_results: int = 10):
    """GET API endpoint with required skills prioritized."""
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    requirements = parse_query_with_gemini(query)
    required_skills = [s.lower().strip() for s in requirements.get("required_skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]
    required_job_levels = [j.lower().strip() for j in requirements.get("job_levels", [])]
    required_languages = [l.lower().strip() for l in requirements.get("languages", [])]
    similar_assessments = retrieve_similar_assessments(query, k=max_results * 2)
    recommendations = []

    for row, similarity_distance in similar_assessments:
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]
        job_levels = [j.lower().strip() for j in str(row["Job Levels"]).split(", ")] if row["Job Levels"] else []
        languages = [l.lower().strip() for l in str(row["Languages"]).split(", ")] if row["Languages"] else []
        job_desc = row["Job Description"].lower()
        scraped_desc = row["Scraped Description"].lower()

        matched_skills = [skill for skill in required_skills if skill in assessment_name or skill in job_desc or skill in scraped_desc or any(skill in t for t in test_types)]
        skill_matches = len(matched_skills)
        skill_score = 0
        if skill_matches == len(required_skills) and required_skills:
            skill_score = 1000
        elif skill_matches > 0:
            skill_score = skill_matches * 200

        secondary_score = 0
        if max_duration and duration != "N/A" and float(duration) <= float(max_duration) * 1.2:
            secondary_score += 50
        elif not max_duration and duration != "N/A" and float(duration) <= 60:
            secondary_score += 25

        if required_test_types and any(test_type in test_types for test_type in required_test_types):
            secondary_score += 40
        elif not required_test_types and any(t in ["knowledge & skills", "ability & aptitude"] for t in test_types):
            secondary_score += 20

        if required_job_levels and any(level in job_levels for level in required_job_levels):
            secondary_score += 30
        elif not required_job_levels and "mid-professional" in job_levels:
            secondary_score += 10

        if required_languages and any(lang in languages for lang in required_languages):
            secondary_score += 20
        elif not required_languages and "english (usa)" in languages:
            secondary_score += 5

        similarity_score = 0 if not EMBEDDINGS_AVAILABLE else (100 - (similarity_distance / np.max(similarity_distance) * 50) if similarity_distance > 0 else 0)
        secondary_score += similarity_score * 0.5

        total_score = skill_score + secondary_score
        if total_score > 0:
            recommendations.append({
                "rank": len(recommendations) + 1,
                "assessment_name": row["Assessment Name"],
                "url": row["URL"],
                "duration": float(row["Duration"]) if row["Duration"] != "N/A" else "N/A",
                "remote_testing_support": row["Remote Testing Support"],
                "adaptive_irt_support": row["Adaptive/IRT Support"],
                "test_type": row["Test Type"],
                "job_levels": row["Job Levels"],
                "languages": row["Languages"],
                "job_description": row["Job Description"],
                "score": total_score
            })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return {"recommendations": recommendations[:max_results] if recommendations else []}

def evaluate_recommendations():
    """Evaluate recommendation performance with recall and MAP metrics."""
    test_queries = [
        {"query": "Hiring Java, Python, SQL developers with .NET Framework, 40 minutes, Mid-Professional, English (USA)",
         "relevant": [".NET Framework 4.5", "Java Programming", "Python Programming", "SQL Server"]},
        {"query": "Research Engineer AI, 60 minutes, Professional Individual Contributor, English (USA)",
         "relevant": ["AI Skills", "Aeronautical Engineering"]}
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
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
        recall_scores.append(recall)
        precision_at_k = 0
        relevant_count = 0
        for j, name in enumerate(recommended_names[:k], 1):
            if name in relevant_set:
                relevant_count += 1
                precision_at_k += relevant_count / j
        ap = precision_at_k / min(k, total_relevant) if min(k, total_relevant) > 0 else 0
        ap_scores.append(ap)

    return np.mean(recall_scores) if recall_scores else 0, np.mean(ap_scores) if ap_scores else 0

def run_streamlit():
    """Run the Streamlit application with an optimized table view."""
    st.title("SHL Assessment Recommendation System :rocket:")
    st.markdown("**Welcome!** Enter a job description or URL to get tailored assessment recommendations, prioritizing required skills. :chart_with_upwards_trend:")
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Input Type", ["Text", "URL"], index=1)
    max_results = st.sidebar.slider("Max Recommendations", 5, 15, 10)
    user_input = st.text_area(
        "Enter Job Description or URL",
        height=150,
        placeholder="E.g., 'Hiring Java, Python, SQL developers with .NET Framework, 40 minutes, Mid-Professional' or a URL",
        value="https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in" if input_type == "URL" else "Hiring Java, Python, and SQL developers with .NET Framework experience, 40 minutes, Mid-Professional."
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
                    table_data = [
                        {
                            "Rank": i + 1,
                            "Assessment Name": row["Assessment Name"],
                            "URL": row["URL"],
                            "Duration (min)": float(row["Duration"]) if row["Duration"] != "N/A" else "N/A",
                            "Remote Testing": row["Remote Testing Support"],
                            "Adaptive/IRT": row["Adaptive/IRT Support"],
                            "Test Type": row["Test Type"],
                            "Job Levels": row["Job Levels"],
                            "Languages": row["Languages"],
                            "Job Description": row["Job Description"][:100] + "..." if len(row["Job Description"]) > 100 else row["Job Description"]  # Truncate long descriptions
                        }
                        for i, (score, row, matched_skills) in enumerate(results)
                    ]
                    recommendations_df = pd.DataFrame(table_data)

                    # Optimized HTML table styling
                    html_table = recommendations_df.to_html(escape=False, index=False)
                    html_table = (
                        html_table
                        .replace('<td>', '<td style="text-align: left; padding: 8px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">')  # Compact cells with ellipsis
                        .replace('<th>', '<th style="background-color: #333333; color: #ffffff; font-weight: bold; padding: 8px; border-bottom: 2px solid #555; text-align: left;">')  # Compact headers
                        .replace('<tr>', '<tr style="background-color: #1a1a1a; color: #ffffff;">')
                        .replace('</tr>', '</tr><tr style="background-color: #1a1a1a; color: #ffffff;">')
                        .replace('<table border="1" class="dataframe">', '<table border="1" class="dataframe" style="width: 100%; max-width: 1200px; background-color: #1a1a1a; color: #ffffff; border-collapse: collapse; overflow-x: auto; display: block;">')  # Fixed width with scroll
                    )
                    for i in range(len(recommendations_df)):
                        html_table = html_table.replace(
                            f'<td>{recommendations_df["URL"][i]}</td>',
                            f'<td style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><a href="{recommendations_df["URL"][i]}" target="_blank" style="color: #1E90FF; text-decoration: underline;">Link</a></td>'
                        )
                    html_table = html_table.replace('</tr>', '</tr><tr style="background-color: #2a2a2a;" onmouseover="this.style.backgroundColor=\'#2a2a2a\';" onmouseout="this.style.backgroundColor=\'#1a1a1a\';">')

                    # Wrap table in a scrollable container
                    st.markdown(
                        """
                        <style>
                        .table-container {
                            max-height: 500px;
                            overflow-y: auto;
                            overflow-x: auto;
                            border: 1px solid #555;
                            margin: 10px 0;
                        }
                        </style>
                        <div class="table-container">
                        """,
                        unsafe_allow_html=True
                    )
                    st.write(html_table, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.write(f"Debug: Number of recommendations: {len(results)}, Top score: {results[0][0] if results else 0}")
                else:
                    st.error("No matching assessments found. Please check the input or dataset.")
                st.subheader("Evaluation Metrics")
                mean_recall, map_k = evaluate_recommendations()
                st.write(f"Mean Recall@5: {mean_recall:.3f}")
                st.write(f"Mean Average Precision @5 (MAP@5): {map_k:.3f}")
            st.subheader("Debug Info")
            st.json(parse_query_with_gemini(user_input))
        else:
            st.warning("Please enter a job description or URL")
    st.sidebar.markdown("""
### 👨‍💻 Yash Singh
**B.Tech CSE, IIIT-Delhi (2022–26)**
📍 New Delhi, India
📧 [singh.yash152004@gmail.com](mailto:singh.yash152004@gmail.com)
📱 +91 9266137288
[🔗 LinkedIn](https://www.linkedin.com/in/yash-singh-a1990025b/)
[💻 GitHub](https://github.com/YSULTRA)
[🏅 LeetCode](https://leetcode.com/u/yash22589/)
[📄 Resume Projects](https://drive.google.com/file/d/1K_bFHoqx0OqGggWToz0Q4TYoAiHBtSa9/view?usp=sharing)
**Developed by Yash Singh** :star2:
""")

async def main():
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    run_streamlit()

if __name__ == "__main__":
    asyncio.run(main())