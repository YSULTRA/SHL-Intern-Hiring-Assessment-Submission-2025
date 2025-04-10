import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to scrape the description from a given URL
def get_description(url):
    # Define a user-agent to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        # Send a GET request to the URL with a 10-second timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the div with class 'col-md-8' that often contains the description
        content_div = soup.find('div', class_='col-md-8')
        if content_div:
            # Extract the first <p> tag within this div
            description_p = content_div.find('p')
            if description_p:
                return description_p.get_text(strip=True)  # Return cleaned text
        return "Description not found"  # Return this if no description is found
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"  # Return error message if request fails

# Wrapper function to return URL and description for threading
def scrape_description(url):
    return url, get_description(url)

# Load the CSV file into a DataFrame
# Replace 'input.csv' with the path to your CSV file
df = pd.read_csv("shl_assessments_updated copy.csv")

# Initialize an empty dictionary to store results
results = {}

# Scrape descriptions concurrently using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all URLs for scraping
    future_to_url = {executor.submit(scrape_description, url): url for url in df["URL"]}
    # Process completed futures as they finish
    for future in as_completed(future_to_url):
        url, description = future.result()
        results[url] = description
        print(f"Scraped {url}")  # Optional: Print progress

# Add the scraped descriptions to a new column in the DataFrame
df["Scraped Description"] = df["URL"].map(results)

# Save the updated DataFrame to a new CSV file
# Replace 'output.csv' with your desired output file name
df.to_csv("output.csv", index=False)

print("CSV file has been updated and saved as 'output.csv'.")