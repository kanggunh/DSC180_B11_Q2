import requests
import time
import pandas as pd
import urllib
from urllib.parse import urlparse, urlunparse
from io import BytesIO
import os
import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
import tempfile
import time
import re
import xml.etree.ElementTree as ET
import json

def get_base_url(url):
    """
    Return the base URL (without query parameters or fragment) from a given URL.
    
    Parameters
    ----------
    url : str
        The URL to extract the base URL from.
    
    Returns
    -------
    str
        The base URL.
    """
    parsed_url = urlparse(url)
    # Reconstruct URL without query parameters and fragment
    return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))

def open_pdf_if_button(driver):
    """
    Checks if the current page has an open button for a PDF, and if so, opens it.

    :param driver: The selenium webdriver to use for scraping.
    :return: True if the PDF was opened, False otherwise.
    """
    try:
        driver.find_element(By.XPATH, "//embed[contains(@type, 'application/pdf')]")
        return True
    except:
        pass
    try:
        for iframe in driver.find_elements(By.TAG_NAME, "iframe"):
            try:
                frame_type = iframe.get_attribute("type")
                if frame_type == "application/pdf":
                    driver.get(iframe.get_attribute("src"))
                    return True
            except:
                print(f"Failed to get link {iframe}")
                return False
    except:
        print("No open button found for current PDF")
    return False

def download_pdf_urls(url, paper_index, driver, download_dir, pdf_dir):
    """
    Downloads PDFs from a given URL and saves them to the specified directory.

    :param url: The URL of the webpage to scrape for PDFs.
    :param paper_index: The index of the paper being scraped.
    :param driver: The selenium webdriver to use for scraping.
    :param download_dir: The temporary directory to download PDFs to.
    :param pdf_dir: The directory to save the final PDFs to.
    :return: The number of PDFs downloaded.
    """
    try:
        driver.get(url)
    except:
        print(f"Failed to get {url}")
        return
    try:
    # Wait for up to 5 seconds for the button to appear and be clickable
        cookie_button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Accept all cookies')]"))
    )
        cookie_button.click()
        print("Accepted cookies.")
    except:
        print("No 'Accept all cookies' button found or it is not clickable.")

    # Identify and exclude links within "references" and "recommended articles" sections
    try:
        recommended_section = driver.find_element(By.XPATH, "//div[@id='recommended-articles']")
        recommended_links = recommended_section.find_elements(By.TAG_NAME, "a")
    except NoSuchElementException:
        recommended_links = []

    try:
        references_section = driver.find_element(By.XPATH, "//ol[@class='references']")
        references_links = references_section.find_elements(By.TAG_NAME, "a")
    except NoSuchElementException:
        references_links = []

    # Combine all links to exclude
    excluded_links = set(recommended_links + references_links)
    pdfs_unique = set()
    pdf_links = []
    pdf_pattern = re.compile(r'(?<!e)\.pdf$|/pdf/|/articlepdf/|/article-pdf/', re.IGNORECASE)
    for link in driver.find_elements(By.TAG_NAME, "a"):
        try:
            href = link.get_attribute("href")
            if link in excluded_links:
                continue
            if href and "scholar.google" not in href and pdf_pattern.search(href):  # selenium cannot download epdfs
                base_url = get_base_url(href)
                if base_url not in pdfs_unique:
                    pdfs_unique.add(base_url)
                    pdf_links.append(href)
        except Exception as e:
            print(e)
            print(f"Failed to get link {link}")
            break
    i = 0
    if len(pdf_links) == 0:
        print(f"No PDF links found for paper {url}")
        return
    downloadable_links_count = 0
    for pdf_link in pdf_links:
        # Ensure each link is a full URL
        pdf_url = pdf_link if pdf_link.startswith('http') else get_base_url(url) + pdf_link
        if "pdf" not in pdf_url: #skips non-pdfs after base url is used
            continue
        try:
            num_of_files_prev = len([f for f in os.listdir(download_dir)])
            curr_url = driver.current_url
            driver.get(pdf_url)
            if curr_url != driver.current_url: # redirected to another page
                open_pdf_if_button(driver)
            time.sleep(1)
            num_of_files_now = len([f for f in os.listdir(download_dir)])
            downloadable_links_count += num_of_files_now > num_of_files_prev
        except Exception as e:
            print(e)
            print(f"Skipping invalid PDF at {pdf_url}")
            continue
    downloaded_pdfs = [f for f in os.listdir(download_dir) if f.endswith('.pdf')]
    while len(downloaded_pdfs) < downloadable_links_count:
        time.sleep(1)
        downloaded_pdfs = [f for f in os.listdir(download_dir) if f.endswith('.pdf')]
    print(f"Downloaded {len(downloaded_pdfs)} PDFs for {url}")
    pdf_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.pdf')]
    i = 1
    for pdf in pdf_files:
        pdf_path = os.path.join(pdf_dir, f'{paper_index}_{i}.pdf')
        output_path = os.path.abspath(os.getcwd() + pdf_path)
        os.rename(pdf, output_path)
        i += 1 

    return len(pdf_files)

def scrape_papers(cross_ref_path, num_papers=10000, pdf_dir="../data/scraping_and_conversion/pdfs"):
    """
    Scrape PDFs from URLs in a CSV file and save them to a specified directory.

    :param cross_ref_path: The path to the CSV file containing the URLs to scrape.
    :param num_papers: The number of papers to scrape. Defaults to 10000.
    :param pdf_dir: The directory to save the PDFs. Defaults to ../data/scraping_and_conversion/pdfs
    """
    download_dir = tempfile.mkdtemp() # os.getcwd() + '/data/pdfs'
    chrome_options = uc.ChromeOptions()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,  # Set download location
        "download.prompt_for_download": False,       # Disable download prompts
        "plugins.always_open_pdf_externally": True   # Download PDFs instead of opening them
    })
    driver = uc.Chrome(options=chrome_options)

    dataset = pd.read_csv(cross_ref_path)
    i = 1
    failed_links = []
    for index, row in dataset.iterrows():
        url = row["URL"]
        print(url)
        num_links = download_pdf_urls(url, i, driver, download_dir, pdf_dir)
        if num_links == 0:
            failed_links.append(row)
        i += 1
        if i == num_papers:
            break
        time.sleep(1)