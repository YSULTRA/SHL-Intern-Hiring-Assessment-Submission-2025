import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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
def get_duration(url, index):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # Mimic a browser
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url} - Status Code: {response.status_code}")
            return {'index': index, 'duration': "N/A"}

        soup = BeautifulSoup(response.content, 'html.parser')
        # Hypothetical selector: Look for text containing "Approximate Completion Time"
        duration_tag = soup.find(lambda tag: tag.name in ['span', 'div', 'p'] and "Approximate Completion Time" in tag.text)
        if duration_tag:
            # Extract the number after "Approximate Completion Time in minutes ="
            duration_text = duration_tag.text
            duration = duration_text.split("Approximate Completion Time in minutes =")[-1].strip().split()[0]
            return {'index': index, 'duration': duration if duration.isdigit() else "N/A"}
        return {'index': index, 'duration': "N/A"}
    except Exception as e:
        print(f"Error fetching duration for {url}: {e}")
        return {'index': index, 'duration': "N/A"}

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

# Use threading to fetch duration for all URLs
duration_list = []
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on your system
    # Create a partial function with the index argument
    fetch_duration = partial(get_duration)
    # Map the function over the URLs with their indices
    duration_list = list(tqdm(
        executor.map(fetch_duration, df['URL'], range(len(df))),
        total=len(df),
        desc="Fetching Duration from URLs"
    ))

# Update DataFrame with fetched durations
for duration_data in duration_list:
    index = duration_data['index']
    df.at[index, 'Duration'] = duration_data['duration']

# Update Test Type to full words for each row
for index, row in df.iterrows():
    df.at[index, 'Test Type'] = convert_test_types(row['Test Type'])

# Save updated data to a new CSV
df.to_csv("shl_assessments_updated.csv", index=False)
print("Updated data saved to shl_assessments_updated.csv")

# Display first few rows
print(df.head())