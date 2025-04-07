import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import streamlit as st
from fastapi import FastAPI
import uvicorn
import requests
from bs4 import BeautifulSoup
import sys
import re
from typing import List, Dict
import json

# Configure Google Gemini API
API_KEY = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=API_KEY)

# Load and preprocess data
df = pd.read_csv('shl_assessments_updated.csv')
df['duration_minutes'] = pd.to_numeric(df['Duration'], errors='coerce')
df['text'] = df['Assessment Name'] + ' - ' + df['Test Type']
df['text_lower'] = df['text'].str.lower()  # For case-insensitive matching

# Use a more advanced embedding model
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# Set up FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Enhanced recommendation logic
def extract_detailed_criteria(query: str) -> Dict:
    """Extract detailed criteria from query using Gemini API with improved prompt."""
    prompt = (
        "Analyze the following query or job description and extract the following in JSON format:\n"
        "1. Maximum duration in minutes (e.g., '60 minutes' -> 60; return 'N/A' if not specified).\n"
        "2. List of key skills or topics (e.g., 'Python, SQL' -> ['Python', 'SQL']; include any mentioned technical or soft skills).\n"
        "3. Preferred test types (e.g., 'Knowledge & Skills'; return empty list if not specified).\n"
        "4. Job level (e.g., 'mid-level', 'senior'; return 'N/A' if unclear).\n"
        "5. Specific competencies (e.g., 'problem-solving', 'teamwork'; return empty list if none specified).\n"
        "6. Purpose (e.g., 'screening', 'development'; return 'N/A' if unclear).\n"
        "Ensure all fields are accurately filled based on the query text.\n"
        f"Query: {query}"
    )
    try:
        response = genai.generate_text(prompt=prompt)
        result = json.loads(response.result.strip())
        return {
            "max_duration": None if result.get("max_duration") == "N/A" else int(result.get("max_duration")),
            "skills": result.get("skills", []),
            "test_types": result.get("test_types", []),
            "job_level": result.get("job_level", "N/A"),
            "competencies": result.get("competencies", []),
            "purpose": result.get("purpose", "N/A")
        }
    except Exception as e:
        print(f"Error extracting criteria: {e}")
        return {"max_duration": None, "skills": [], "test_types": [], "job_level": "N/A", "competencies": [], "purpose": "N/A"}

def scrape_job_description(url: str) -> str:
    """Scrape and preprocess job description from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'li', 'div', 'span']) if len(tag.get_text(strip=True).split()) > 5]
        job_text = ' '.join(content)
        job_text = re.sub(r'\s+', ' ', job_text).strip()
        if len(job_text.split()) > 500:
            prompt = f"Summarize the following job description into key requirements in 200 words or less:\n{job_text}"
            response = genai.generate_text(prompt=prompt)
            return response.result.strip()
        return job_text
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return ""

def calculate_relevance(row: pd.Series, criteria: Dict, similarity_score: float) -> tuple[float, str]:
    """Calculate relevance score with stronger emphasis on skill matches."""
    score = similarity_score * 0.4  # Reduce weight of semantic similarity
    explanation = ["Based on query context"]

    # Skill matches (higher weight)
    skills = [s.lower() for s in criteria["skills"]]
    if skills:
        assessment_text = row['text_lower']
        skill_matches = sum(1 for skill in skills if skill in assessment_text)
        score += 0.3 * skill_matches  # Higher boost for skills
        if skill_matches:
            matched_skills = [s for s in skills if s in assessment_text]
            explanation.append(f"Matches skills: {', '.join(matched_skills)}")
        elif any(skill in assessment_text for skill in ["coding", "programming", "development"]):
            score += 0.1  # Fallback for related terms
            explanation.append("Matches related programming context")

    # Test type matches
    test_types = criteria["test_types"]
    if test_types and row['Test Type'] in test_types:
        score += 0.2
        explanation.append(f"Matches test type: {row['Test Type']}")

    # Duration compliance
    max_duration = criteria["max_duration"]
    if max_duration and row['duration_minutes'] <= max_duration:
        score += 0.2
        explanation.append(f"Within duration: {row['Duration']} <= {max_duration} min")
    elif max_duration and pd.isna(row['duration_minutes']):
        score -= 0.1  # Penalty for unknown duration
        explanation.append("Duration unknown")

    # Job level and competencies (minor boosts)
    job_level = criteria["job_level"].lower()
    if job_level != "n/a" and job_level in row['text_lower']:
        score += 0.1
        explanation.append(f"Matches job level: {job_level}")

    competencies = [c.lower() for c in criteria["competencies"]]
    if competencies:
        comp_matches = sum(1 for comp in competencies if comp in row['text_lower'])
        score += 0.15 * comp_matches
        if comp_matches:
            explanation.append(f"Matches competencies: {', '.join([c for c in competencies if c in row['text_lower']])}")

    return min(score, 1.0), "; ".join(explanation)

def get_recommendations(query_text: str, is_url: bool = False) -> List[Dict]:
    """Retrieve and rank SHL assessment recommendations."""
    original_query = query_text
    if is_url:
        query_text = scrape_job_description(query_text)
        if not query_text:
            return []

    criteria = extract_detailed_criteria(original_query if is_url else query_text)
    max_duration = criteria["max_duration"]

    query_embedding = model.encode([query_text])
    D, I = index.search(query_embedding, k=50)
    distances = D[0]
    candidates = df.iloc[I[0]].copy()

    max_distance = max(distances) if distances.max() > 0 else 1
    candidates['similarity'] = 1 - (distances / max_distance)

    candidates[['relevance', 'explanation']] = candidates.apply(
        lambda row: pd.Series(calculate_relevance(row, criteria, row['similarity'])), axis=1
    )

    if max_duration is not None:
        candidates = candidates[
            (candidates['duration_minutes'].notna()) &
            (candidates['duration_minutes'] <= max_duration)
        ]

    top_10 = candidates.sort_values(by='relevance', ascending=False).head(10)

    if len(top_10) < 1 and len(candidates) > 0:  # Ensure at least 1 recommendation
        top_10 = candidates.sort_values(by='relevance', ascending=False).head(1)

    output = top_10[[
        'Assessment Name', 'URL', 'Remote Testing Support',
        'Adaptive/IRT Support', 'Duration', 'Test Type', 'relevance', 'explanation'
    ]].rename(columns={'relevance': 'Confidence Score'}).to_dict(orient='records')

    return output

# Streamlit UI
def run_streamlit():
    st.set_page_config(page_title="SHL Assessment Recommendation System", layout="wide")
    st.title("SHL Assessment Recommendation System")
    st.markdown("**Find the perfect SHL assessments for your hiring needs.** Enter a query, job description, or URL below.")

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form(key='input_form'):
            input_text = st.text_area(
                "Job Description, Query, or URL",
                height=150,
                placeholder="e.g., 'I am hiring for Java developers who can collaborate effectively, completed in 40 minutes'"
            )
            is_url = st.checkbox("Input is a URL")
            submit_button = st.form_submit_button(label="Get Recommendations", type="primary")
    with col2:
        st.info("**Tips:**\n- Specify skills (e.g., Python, SQL), duration, or test types.\n- Use a job posting URL for automatic analysis.")

    if submit_button and input_text:
        with st.spinner("Analyzing and generating recommendations..."):
            recommendations = get_recommendations(input_text, is_url)

        st.write(f"**Input Text:** {input_text}")  # Show input for debugging

        if recommendations:
            # Display criteria
            criteria = extract_detailed_criteria(input_text if not is_url else scrape_job_description(input_text))
            with st.expander("Extracted Criteria", expanded=True):
                st.write(f"**Max Duration:** {criteria['max_duration'] if criteria['max_duration'] else 'N/A'} minutes")
                st.write(f"**Skills:** {', '.join(criteria['skills']) if criteria['skills'] else 'None specified'}")
                st.write(f"**Test Types:** {', '.join(criteria['test_types']) if criteria['test_types'] else 'None specified'}")
                st.write(f"**Job Level:** {criteria['job_level']}")
                st.write(f"**Competencies:** {', '.join(criteria['competencies']) if criteria['competencies'] else 'None specified'}")
                st.write(f"**Purpose:** {criteria['purpose']}")

            # Recommendations Table
            st.subheader("Recommended Assessments")
            df_display = pd.DataFrame(recommendations)
            df_display['Assessment Name'] = df_display.apply(
                lambda row: f'<a href="{row["URL"]}" target="_blank">{row["Assessment Name"]}</a>', axis=1
            )
            df_display['Confidence Score'] = df_display['Confidence Score'].apply(lambda x: f"{x:.2%}")
            st.write(
                df_display[[
                    'Assessment Name', 'Remote Testing Support', 'Adaptive/IRT Support',
                    'Duration', 'Test Type', 'Confidence Score', 'explanation'
                ]].rename(columns={'explanation': 'Why Recommended'}).to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

            # Downloadable report
            report = df_display.to_csv(index=False)
            st.download_button(
                label="Download Recommendations",
                data=report,
                file_name="shl_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.warning("No recommendations found. Check if the skills or criteria match available assessments.")

# FastAPI Endpoint
app = FastAPI(title="SHL Assessment Recommendation API")

@app.get("/recommend")
async def recommend(query: str, is_url: bool = False):
    """Get SHL assessment recommendations for a given query or URL."""
    recommendations = get_recommendations(query, is_url)
    return {"recommendations": recommendations}

# Main execution
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_streamlit()