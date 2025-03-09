import os
import requests
import grobid_tei_xml
import pandas as pd

def convert_pdf_to_xml(pdf_dir='../data/scraping_and_conversion/pdfs', xml_dir='../data/scraping_and_conversion/xmls'):
    """
    Converts a PDF file to XML format using the GROBID API.

    :param pdf_path: The path to the PDF file to be converted.
    :param xml_path: The path where the XML file will be saved.
    """
    grobid_url = "http://localhost:8070/api/processFulltextDocument"
    xml_names = os.listdir("data/xmls")

    for pdf_file in os.listdir(pdf_dir):
        #only looks at pdf files
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(pdf_path)
            #doe not convert already converted files
            if pdf_file.replace('.pdf', '.xml') in xml_names:
                continue
            with open(pdf_path, 'rb') as file:
                #GROBID must be running on port 8070 for this to work
                response = requests.post(
                    grobid_url,
                    files={'input': file},
                    headers={'Accept': 'application/xml'}
                )
                
                if response.status_code == 200:
                    xml_file_path = os.path.join(xml_dir, pdf_file.replace('.pdf', '.xml'))
                    with open(xml_file_path, 'w', encoding='utf-8') as xml_file:
                        xml_file.write(response.text)
                    print(f"Converted {pdf_file} to {xml_file_path}")
                else:                
                    print(f"Failed to convert {pdf_file}. Status code: {response.status_code}")
                    print(response.text)


def parse_grobid_xml(file_path):
    """
    Parse a GROBID XML file and return a string containing
    the paper's title, abstract, and body.

    :param file_path: The path to the GROBID XML file to be parsed.
    :return: A string containing the paper's title, abstract, and body.
    """
    with open(file_path, "r", encoding="utf-8") as xml_file:
        try:
            doc = grobid_tei_xml.parse_document_xml(xml_file.read())
        except:
            return ""
        title = "Title: " + doc.header.title if doc.header.title else ""
        abstract = doc.abstract or ""
        body = doc.body or ""
        index = file_path.split("/")[-1].split(".")[0]
        return f"Paper #: {index}\n{title}\n{abstract}\n{body}\n" #title, abstract, body


def convert_grobid_xml_to_csv(output_file, xml_dir='../data/scraping_and_conversion/xmls', previous_batch_path=None):
    """
    Convert a directory of GROBID XML files to a CSV file by merging
    the text from multiple XML files into a single column in the CSV.

    :param output_file: The path to the CSV file to be written.
    :param xml_dir: The directory containing the GROBID XML files.
        Defaults to '../data/scraping_and_conversion/xmls'.
    :param previous_batch_path: The path to a CSV file containing a previous
        batch of data. If provided, the two datasets will be merged.
    :return: The output DataFrame.
    """
    crossref_df = pd.read_csv("data/crossref_data.csv")
    crossref_df["text"] = None
    for index, _ in crossref_df.iterrows():
        i = 1
        added_count = 0
        paper_text = ""
        while added_count < 3 and i <= 6: #prevents more than 3 files from being added, up to 6 accounts for grobid failures to generate xml
            file_path = f"{xml_dir}/{index}_{i}.xml"
            if os.path.exists(file_path):
                paper_text += parse_grobid_xml(file_path)
                added_count += 1
            i += 1
        crossref_df.at[index, "text"] = paper_text
    output_df = crossref_df[crossref_df["text"].str.len() > 0]
    unique_dois = output_df.groupby('DOI').first()
    if previous_batch_path is not None:
        previous_batch = pd.read_csv(previous_batch_path)
        unique_dois = pd.concat([previous_batch, unique_dois], ignore_index=True)
    unique_dois.to_csv(output_file)
    return unique_dois