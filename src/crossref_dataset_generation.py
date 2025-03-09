import requests, time
import pandas as pd


def generate_crossref_dataset(save_path, most_recent_year=2024):
    """
    Generate a dataset of 50,000 DOIs for perovskite solar cells using the Crossref API.
    10,000 DOIs per year are retreived for the last 5 years.
    
    :param most_recent_year: The year to start the search from (going back 5 years). Default is 2024.
    """
    query = "perovskite solar halide passivation"
    base_url = f"https://api.crossref.org/works"

    rows_per_request = 1000
    all_dois = []

    for i in range(most_recent_year, most_recent_year - 5, -1):
        offset = 0
        while offset < 10000:
            url = f"{base_url}?query={query}&rows={rows_per_request}&offset={offset}&filter=from-pub-date:{i},until-pub-date:{i}"
            response = requests.get(url)

            if response.status_code != 200:
                print(f"Failed request at offset {offset}: {response.status_code}")
                break

            data = response.json()
            items = data['message']['items']

            if not items:  # Stop when there are no more results
                break

            for item in items:
                if 'DOI' in item:
                    all_dois.append(item)

            print(f"Fetched {len(items)} records (Offset: {offset})")
            offset += rows_per_request
            time.sleep(1)  # Be polite and avoid rate limits
    print(f"Saving {len(all_dois)} DOIs to {save_path}")
    df = pd.DataFrame(all_dois)
    df.to_csv(save_path, index=False)
    print("DOIs saved successfully!")