import sys
from src.crossref_dataset_generation import generate_crossref_dataset
from src.scraping import scrape_papers
from src.pdf_conversion import convert_pdf_to_xml, convert_grobid_xml_to_csv
from src.classification import apply_keyword_classification, train_classification_models
from src.rag_filtering import filter_with_rag
from src.extraction import run_extraction, convert_csv_to_json
from src.extraction_evaluation import evaluate_extraction
from src.format_extraction import ensure_json_format, format_passivators

def run_scraping_and_conversion():
    crossref_save_path = '../data/scraping_and_conversion/crossref_data.csv'
    papers_output_path = '../data/scraped_and_conversion/scraped_papers.csv'
    generate_crossref_dataset(crossref_save_path, 2024)
    scrape_papers(crossref_save_path)
    convert_pdf_to_xml()
    convert_grobid_xml_to_csv(papers_output_path)

def run_classification():
    train_classification_models()
    apply_keyword_classification()

def run_extraction_for_eval():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    annotations_path = "../data/annotations/150_papers.csv"
    rag_output_path = filter_with_rag(annotations_path)
    extraction_csv_path = run_extraction(model_name=model_name, tokenizer_name=model_name, data_path=rag_output_path)
    extraction_json_path = f"../data/extraction_eval/{model_name.split('/')[-1]}.json"
    convert_csv_to_json(extraction_csv_path, extraction_json_path)
    formatted_path = ensure_json_format(extraction_json_path)
    format_passivators(formatted_path)


def run_full_extraction():
    relevant_papers_path = "../data/classification/relevant_papers.csv"
    rag_output_path = filter_with_rag(relevant_papers_path)
    extraction_csv_path = run_extraction(rag_output_path)
    extraction_json_path = "../data/extraction_final/final_extraction.json"
    convert_csv_to_json(extraction_csv_path, extraction_json_path)
    formatted_path = ensure_json_format(extraction_json_path)
    format_passivators(formatted_path)

def run_prediction():
    #TODO: Add the rest of the prediction code

def run_full_pipeline():
    run_scraping_and_conversion()
    run_classification()
    run_full_extraction()

if __name__ == "__main__":
    args = sys.argv[1:]

    if 'scraping_and_conversion' in args:
        run_scraping_and_conversion()
    if 'classification' in args:
        run_classification()
    if 'finetuning' in args:
        #TODO: Add the rest of the fine tuning code
    if 'extraction_evaluation' in args:
        run_extraction_for_eval()
    if 'extraction' in args:
        run_full_extraction()
    if 'prediction' in args:
        run_prediction()
    elif 'all' in args or len(args) == 0:
        run_scraping_and_conversion()
        run_classification()
        run_full_extraction()
        #TODO: Add the rest of the all code