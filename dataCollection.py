import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time

# Base URL and pagination parameters
base_url = "https://www.shl.com/solutions/products/product-catalog/"
start_values = range(0, 373, 12)  # From 0 to 372, step by 12

# List to store all assessment data
assessments = []

# Function to scrape a single page
def scrape_page(start):
    url = f"{base_url}?start={start}&type=1&type=1"
    headers = {'User-Agent': 'Mozilla/5.0'}  # Mimic a browser to avoid blocks
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch {url} - Status Code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    table_div = soup.find('div', class_='custom__table-responsive')

    if not table_div:
        print(f"No table div found on {url}")
        return

    table = table_div.find('table')
    if not table:
        print(f"No table found inside div on {url}")
        return

    # Try to get rows from <tbody>, fall back to <table> if <tbody> is missing
    tbody = table.find('tbody')
    if tbody:
        rows = tbody.find_all('tr')[1:]  # Skip header row
    else:
        rows = table.find_all('tr')[1:]  # Directly from table if no tbody

    if not rows:
        print(f"No rows found in table on {url}")
        return

    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 4:
            continue

        # Extract Assessment Name and URL
        name_link = cols[0].find('a')
        if not name_link:
            continue
        assessment_name = name_link.text.strip()
        assessment_url = "https://www.shl.com" + name_link['href']

        # Remote Testing Support
        remote_testing = "Yes" if cols[1].find('span', class_='catalogue__circle -yes') else "No"

        # Adaptive/IRT Support
        adaptive_irt = "Yes" if cols[2].find('span', class_='catalogue__circle -yes') else "No"

        # Test Type (from keys)
        test_type_spans = cols[3].find_all('span', class_='product-catalogue__key')
        test_types = [span.text.strip() for span in test_type_spans]
        test_type = ", ".join(test_types) if test_types else "Unknown"

        # Duration (placeholder; fetch from individual page if needed)
        duration = "N/A"

        # Append to list
        assessments.append({
            "Assessment Name": assessment_name,
            "URL": assessment_url,
            "Remote Testing Support": remote_testing,
            "Adaptive/IRT Support": adaptive_irt,
            "Duration": duration,
            "Test Type": test_type
        })

# Scrape all pages with progress bar
for start in tqdm(start_values, desc="Scraping SHL Catalog Pages"):
    scrape_page(start)
    time.sleep(1)  # Avoid overwhelming the server

# Save to CSV
df = pd.DataFrame(assessments)
df.to_csv("shl_assessments.csv", index=False)
print("Data saved to shl_assessments.csv")

# Display first few rows
print(df.head())