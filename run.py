import sys
from src.crossref_dataset_generation import generate_crossref_dataset
from src.scraping import scrape_papers
from src.pdf_conversion import convert_pdf_to_xml, convert_grobid_xml_to_csv
from src.classification import apply_keyword_classification, train_classification_models
from src.rag_filtering import filter_with_rag
from src.extraction import run_extraction, convert_csv_to_json
from src.format_extraction import ensure_json_format, format_passivators, create_df_from_json
from src.db_processing import clean_and_merge_db
from src.prediction import run_prediction
from src.finetuning import finetune_extraction_model

def run_scraping_and_conversion():
    """
    Runs the scraping and conversion pipeline, which includes generating the crossref dataset, scraping papers from the crossref dataset, converting scraped PDFs to XML, and converting the XML to CSV.

    The Crossref dataset is saved to '../data/scraping_and_conversion/crossref_data.csv' and the scraped papers are saved to '../data/scraped_and_conversion/scraped_papers.csv'.
    """
    crossref_save_path = '../data/scraping_and_conversion/crossref_data.csv'
    papers_output_path = '../data/scraped_and_conversion/scraped_papers.csv'
    generate_crossref_dataset(crossref_save_path, 2024)
    scrape_papers(crossref_save_path)
    convert_pdf_to_xml()
    convert_grobid_xml_to_csv(papers_output_path)

def run_classification():
    """
    Runs the classification pipeline, which includes training the classification models and applying the trained models to classify the papers as relevant or not.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    train_classification_models()
    apply_keyword_classification()

def run_extraction_for_eval():
    """
    Runs the extraction pipeline for evaluation, taking the 150 papers annotation and applying RAG to filter out irrelevant papers, then running the extraction model and converting the output to JSON format and formatting the passivators.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_name = "../models/Llama-PSC-Extractor-8B-8bit-Schema-2"
    annotations_path = "../data/annotations/150_papers.csv"
    rag_output_path = filter_with_rag(annotations_path)
    extraction_csv_path = run_extraction(model_name=model_name, tokenizer_name=model_name, data_path=rag_output_path)
    extraction_json_path = f"../data/extraction_eval/{model_name.split('/')[-1]}.json"
    convert_csv_to_json(extraction_csv_path, extraction_json_path)
    formatted_path = ensure_json_format(extraction_json_path)
    format_passivators(formatted_path)


def run_full_extraction():
    """
    Runs the full extraction pipeline, taking the relevant papers from the classification step and outputting the final prediction dataset.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_name = "../models/Llama-PSC-Extractor-8B-8bit-Schema-2"
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    relevant_papers_path = "../data/classification/relevant_papers.csv"
    rag_output_path = filter_with_rag(relevant_papers_path)
    extraction_csv_path = run_extraction(rag_output_path, model_name=model_name, tokenizer_name=tokenizer_name)
    extraction_json_path = "../data/extraction_final/final_extraction.json"
    convert_csv_to_json(extraction_csv_path, extraction_json_path)
    formatted_path = ensure_json_format(extraction_json_path)
    format_passivators(formatted_path)
    prediction_df_path = create_df_from_json(formatted_path)
    annotated_expanded_path = "../data/prediction/annotations_expanded.csv"
    merged_path = "../data/prediction/prediction_dataset.csv"
    clean_and_merge_db(prediction_df_path, annotated_expanded_path, merged_path)

def run_full_pipeline():
    run_scraping_and_conversion()
    run_classification()
    run_full_extraction()

if __name__ == "__main__":
    args = sys.argv[1:]
    prediction_df_path = "../data/prediction/prediction_dataset.csv"
    if 'scraping_and_conversion' in args:
        run_scraping_and_conversion()
    if 'classification' in args:
        run_classification()
    if 'finetuning' in args:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        training_data_path = "../data/finetuning/chunked_training_schema2.csv"
        finetune_extraction_model(model_name=model_name, training_data_path=training_data_path)
    if 'extraction_evaluation' in args:
        run_extraction_for_eval()
    if 'extraction' in args:
        run_full_extraction()
    if 'prediction' in args:
        run_prediction(prediction_df_path)
    elif 'all' in args or len(args) == 0:
        run_scraping_and_conversion()
        run_classification()
        run_full_extraction()
        run_prediction(prediction_df_path)