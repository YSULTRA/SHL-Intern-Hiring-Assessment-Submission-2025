import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time

# Mapping of test type abbreviations to full words
test_type_mapping = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

# Function to extract duration from individual assessment page
def get_duration(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # Mimic a browser
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url} - Status Code: {response.status_code}")
            return "N/A"

        soup = BeautifulSoup(response.content, 'html.parser')
        # Hypothetical selector: Look for text containing "Approximate Completion Time"
        duration_tag = soup.find(lambda tag: tag.name in ['span', 'div', 'p'] and "Approximate Completion Time" in tag.text)
        if duration_tag:
            # Extract the number after "Approximate Completion Time in minutes ="
            duration_text = duration_tag.text
            duration = duration_text.split("Approximate Completion Time in minutes =")[-1].strip().split()[0]
            return duration if duration.isdigit() else "N/A"
        return "N/A"
    except Exception as e:
        print(f"Error fetching duration for {url}: {e}")
        return "N/A"

# Function to convert test types to full words
def convert_test_types(test_types_str):
    if test_types_str == "Unknown" or not test_types_str:
        return "Unknown"
    types = test_types_str.split(", ")
    full_types = [test_type_mapping.get(t, t) for t in types]
    return ", ".join(full_types)

# Load existing CSV data
try:
    df = pd.read_csv("shl_assessments.csv")
except FileNotFoundError:
    print("Error: shl_assessments.csv not found. Please run the initial scraping script first.")
    exit()

# Update Duration and Test Type for each row
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating Duration and Test Types"):
    # Update Duration
    duration = get_duration(row['URL'])
    df.at[index, 'Duration'] = duration

    # Update Test Type to full words
    df.at[index, 'Test Type'] = convert_test_types(row['Test Type'])

# Save updated data to a new CSV
df.to_csv("shl_assessments_updated.csv", index=False)
print("Updated data saved to shl_assessments_updated.csv")

# Display first few rows
print(df.head())