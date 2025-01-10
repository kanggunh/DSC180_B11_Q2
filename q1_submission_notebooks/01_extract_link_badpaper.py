"""Run this python file to create a dataframe of links for all irrelevent papers in perovskite research"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

"""Second link has HPPT 403 error, so use Selenium to go around this issues"""

driver = webdriver.Chrome()
url_2 = 'https://www.sciencedirect.com/science/article/pii/S2590238524001711'

# Open the research paper URL
driver.get(url_2)

# Locate the references section in this url
references = driver.find_elements(By.XPATH, "//ol[@class='references']//a[contains(@class, 'anchor anchor-primary')]")

# Extract the links
link_list = [ref.get_attribute('href') for ref in references if ref.get_attribute('href')]


print("Extracted links:", link_list)

# Close the driver
driver.quit()

"""Access the first link by beautiful soup"""
#Fetching the webpage content using url
url_1 = 'https://www.nature.com/articles/s41467-019-08507-4'
response_1 = requests.get(url_1)

# Check if the page is fetched successfully
if response_1.status_code == 200:
    # Parse the page content using Beautiful Soup
    soup = BeautifulSoup(response_1.content, 'html.parser')

    # Find the class where all the reference is stored as a list
    references = soup.find_all('li', class_='c-article-references__item')

    #Initiallized list to store all the URL in the reference
    for ref in references:
        link = ref.find('a', href=True)
        if link:
            # print(link['href'])
            link_list.append(link['href'])
    

else:
    print(f"Failed to fetch the page. Status code: {response_1.status_code}")

link_set = set(link_list)
link_list_unique = list(link_set)
#Store this result in pandas df
bad_paperdf = pd.DataFrame({
"link": link_list_unique
})


#Export this pdf as a csv file
bad_paperdf.to_csv("../Irrelevent_paper.csv", index=False)
