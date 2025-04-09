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
os.environ['HF_HUB_OFFLINE'] = '1'
API_KEY = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Initialize Streamlit application
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model(max_retries=3, delay=5):
    """Load the SentenceTransformer model with retry logic."""
    try:
        from sentence_transformers import SentenceTransformer

        # First try to download the model if not available locally
        model_name = "all-MiniLM-L6-v2"
        try:
            model = SentenceTransformer(model_name)
            st.success(f"Successfully loaded SentenceTransformer model: {model_name}")
            return model
        except Exception as download_error:
            st.warning(f"Failed to download model: {download_error}. Trying local cache...")

            # Define possible model directories
            possible_dirs = [
                os.path.join(os.path.dirname(__file__), "models", model_name),
                os.path.join(os.getcwd(), "models", model_name),
                os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers", model_name)
            ]

            # Try each possible directory
            for model_dir in possible_dirs:
                try:
                    if os.path.exists(model_dir):
                        model = SentenceTransformer(model_dir)
                        st.success(f"Successfully loaded model from: {model_dir}")
                        return model
                except Exception as e:
                    st.warning(f"Failed to load from {model_dir}: {e}")
                    continue

            st.warning(f"Failed to load SentenceTransformer model after checking all locations. Running without embeddings.")
            return None

    except ImportError as e:
        st.error(f"Failed to import SentenceTransformer due to: {e}. Running without embeddings.")
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
    """Load and filter the assessment dataset."""
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

@st.cache_data
def setup_vector_database(_df):
    """Set up the FAISS vector database with embeddings."""
    if not EMBEDDINGS_AVAILABLE or embedding_model is None:
        return None, [], []

    try:
        descriptions = [f"{row['Assessment Name']} {row.get('URL', '')} {' '.join(row['Assessment Name'].lower().split())}"
                       for _, row in _df.iterrows()]
        embeddings = embedding_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) if faiss else None
        if index:
            index.add(embeddings)
        return index, descriptions, embeddings
    except Exception as e:
        st.error(f"Error setting up vector database: {e}")
        return None, [], []

index, descriptions, embeddings = setup_vector_database(df)

def extract_text_from_url_threaded(url, result_queue):
    """Extract text from a URL using threading."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try common description-containing elements
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
    except Exception as e:
        result_queue.put(f"Unexpected error processing URL {url}: {e}")

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
        context = "\n".join([f"Assessment: {assess['Assessment Name']}, Type: {assess['Test Type']}, Duration: {assess['Duration']}"
                            for assess, _ in similar_assessments])

        prompt = f"""
        You are an expert in job assessment analysis. Analyze the following query and context to extract:
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

                # Fallback skill extraction if Gemini didn't find any
                if not result.get("required_skills"):
                    result["required_skills"] = [skill for skill in ["java", "python", "sql", ".net framework"] if skill in query.lower()]

                # Fallback test type if none identified
                if not result.get("test_types"):
                    result["test_types"] = ["Knowledge & Skills"] if any(skill in query.lower() for skill in ["java", "python", "sql", ".net"]) else []

                # Extract duration from text if not provided by Gemini
                if not result.get("max_duration") and "minutes" in query.lower():
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
    """Cache and parse query using Gemini API."""
    result_queue = Queue()
    thread = threading.Thread(target=parse_query_with_gemini_threaded, args=(query, result_queue))
    thread.start()
    thread.join(timeout=15)
    return result_queue.get() if not thread.is_alive() else {"required_skills": [], "max_duration": None, "test_types": []}

@st.cache_data
def recommend_assessments(query, max_results=10):
    """Generate assessment recommendations based on query."""
    requirements = parse_query_with_gemini(query)
    required_skills = [s.lower().strip() for s in requirements.get("required_skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]
    similar_assessments = retrieve_similar_assessments(query, k=max_results * 2)
    recommendations = []

    for row, similarity_distance in similar_assessments:
        score = 0
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]

        # Skill matching scoring
        skill_matches = sum(1 for skill in required_skills if skill in assessment_name or any(skill in t.lower() for t in test_types))
        if skill_matches == len(required_skills):
            score += 100
        elif skill_matches > 0:
            score += skill_matches * 60

        # Duration scoring
        if max_duration and duration != "N/A":
            try:
                if float(duration) <= float(max_duration) * 1.2:
                    score += 30
            except ValueError:
                pass
        elif not max_duration and duration != "N/A":
            try:
                if float(duration) <= 60:
                    score += 15
            except ValueError:
                pass

        # Test type scoring
        if required_test_types and any(test_type in ", ".join(test_types) for test_type in required_test_types):
            score += 50
        elif not required_test_types and any(t in ["knowledge & skills", "ability & aptitude"] for t in test_types):
            score += 20

        # Similarity scoring (if embeddings available)
        if EMBEDDINGS_AVAILABLE and similarity_distance > 0 and np is not None:
            try:
                similarity_score = 100 - (similarity_distance / np.max(similarity_distance) * 50)
                score += similarity_score * 0.3
            except:
                pass

        if score > 0:
            recommendations.append((score, row))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations[:max_results] if recommendations else []

# API Endpoint
@app.get("/recommend")
async def get_recommendations(query: str, max_results: int = 10):
    """GET API endpoint to retrieve assessment recommendations in JSON format."""
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    # Process the query (similar to Streamlit logic)
    requirements = parse_query_with_gemini(query)
    required_skills = [s.lower().strip() for s in requirements.get("required_skills", [])]
    max_duration = requirements.get("max_duration")
    required_test_types = [t.lower().strip() for t in requirements.get("test_types", [])]
    similar_assessments = retrieve_similar_assessments(query, k=max_results * 2)
    recommendations = []

    for row, similarity_distance in similar_assessments:
        score = 0
        assessment_name = row["Assessment Name"].lower()
        duration = row["Duration"]
        test_types = [t.lower().strip() for t in row["Test Type"].split(", ")]

        skill_matches = sum(1 for skill in required_skills if skill in assessment_name or any(skill in t.lower() for t in test_types))
        if skill_matches == len(required_skills):
            score += 100
        elif skill_matches > 0:
            score += skill_matches * 60

        if max_duration and duration != "N/A":
            try:
                if float(duration) <= float(max_duration) * 1.2:
                    score += 30
            except ValueError:
                pass
        elif not max_duration and duration != "N/A":
            try:
                if float(duration) <= 60:
                    score += 15
            except ValueError:
                pass

        if required_test_types and any(test_type in ", ".join(test_types) for test_type in required_test_types):
            score += 50
        elif not required_test_types and any(t in ["knowledge & skills", "ability & aptitude"] for t in test_types):
            score += 20

        if EMBEDDINGS_AVAILABLE and similarity_distance > 0 and np is not None:
            try:
                similarity_score = 100 - (similarity_distance / np.max(similarity_distance) * 50)
                score += similarity_score * 0.3
            except:
                pass

        if score > 0:
            recommendations.append({
                "rank": len(recommendations) + 1,
                "assessment_name": row["Assessment Name"],
                "url": row["URL"],
                "duration": float(row["Duration"]) if row["Duration"] != "N/A" else "N/A",
                "remote_testing_support": row["Remote Testing Support"],
                "adaptive_irt_support": row["Adaptive/IRT Support"],
                "test_type": row["Test Type"],
                "score": score
            })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return {"recommendations": recommendations[:max_results] if recommendations else []}

def evaluate_recommendations():
    """Evaluate recommendation performance with recall and MAP metrics."""
    test_queries = [
        {"query": "Hiring Java, Python, SQL developers with .NET Framework, 40 minutes", "relevant": ["Java Programming", "Python Programming", "SQL Server", ".NET Framework 4.5"]},
        {"query": "Research Engineer AI, 60 minutes", "relevant": ["AI Fundamentals", "Research Skills"]}
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
    """Run the Streamlit application for assessment recommendations."""
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
                    table_data = [
                        {
                            "Rank": i + 1,
                            "Assessment Name": row["Assessment Name"],
                            "URL": row["URL"],
                            "Duration (min)": float(row["Duration"]) if row["Duration"] != "N/A" else "N/A",
                            "Remote Testing Support": row["Remote Testing Support"],
                            "Adaptive/IRT Support": row["Adaptive/IRT Support"],
                            "Test Type": row["Test Type"]
                        }
                        for i, (_, row) in enumerate(results)
                    ]

                    recommendations_df = pd.DataFrame(table_data)
                    html_table = recommendations_df.to_html(escape=False, index=False)

                    # Style the HTML table
                    html_table = (
                        html_table
                        .replace('<td>', '<td style="text-align: left; padding: 10px;">')
                        .replace('<th>', '<th style="background-color: #333333; color: #ffffff; font-weight: bold; padding: 10px; border-bottom: 2px solid #555;">')
                        .replace('<tr>', '<tr style="background-color: #1a1a1a; color: #ffffff;">')
                        .replace('</tr>', '</tr><tr style="background-color: #1a1a1a; color: #ffffff;">')
                        .replace('<table border="1" class="dataframe">', '<table border="1" class="dataframe" style="background-color: #1a1a1a; color: #ffffff; width: 100%;">')
                    )

                    # Convert URLs to clickable links
                    for i in range(len(recommendations_df)):
                        html_table = html_table.replace(
                            f'<td>{recommendations_df["URL"][i]}</td>',
                            f'<td><a href="{recommendations_df["URL"][i]}" target="_blank" style="color: #1E90FF; text-decoration: underline;">Link</a></td>'
                        )

                    # Add hover effects
                    html_table = html_table.replace('</tr>', '</tr><tr style="background-color: #2a2a2a;" onmouseover="this.style.backgroundColor=\'#2a2a2a\';" onmouseout="this.style.backgroundColor=\'#1a1a1a\';">')

                    st.write(html_table, unsafe_allow_html=True)
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

    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    asyncio.run(main())