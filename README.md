# Perovskite Solar Cells: Literature Mining Project

### Folder Structure
```
DSC180_B11
└───data                                   # Collection of all data
│    │   biocs/                                
│    │   model_results/                        
│    │   txts/                               
│    │   xmls/                                 
│    │   .
│    │   .
│    │   .
│    │   .
│    └───training_data.csv                 
│
└───images
│    │   classification_compare.png        # Model comparison visualization
│    │   pipeline.png                      # Workflow diagram
│    └───q2_timeline.png                   # Project timeline visualization
│
└───models
│    │   llama-3.2-3b-it-Perovskite-PaperExtractor/
│    │   prediction_model                  # work in progress
│    └───scibert_psc_ner_model/                
│
└───q2_submission_notebooks
│    │   00_crossref_scraping.ipynb           
│    │   01_Scrapint_texts.ipynb            
│    │   02_TF-IDF_vectorizer_and_models.ipynb
│    │   03_docling.ipynb                  
│    │   04_pretrained_model_extraction.ipynb                   
│    │   05_chatextract_openai.py    
│    │   06_chunked_training_creation.ipynb              
│    │   07_finetuning_peft.ipynb   
│    │   08_finetuning_test.ipynb                     
│    └───09_evaluation_final.ipynb          
│
└───README.md                                  
└───requirements.txt                       
└───run.py                                     

```

### Introduction
This project aims to optimize the discovery of small molecule that improve the stability of perovskite solar cells. By leveraging literature mining, graph-based molecular representations, and machine learning models, we seek to identify patterns that lead to successful molecules and generate deeper insights into perovskite solar cell performance. The current methods rely heavily on Edisonian experimentation, which is inefficient. Our goal is to automate and streamline this discovery process using data-driven techniques, focusing on the vast body of research already conducted in this field.

### Objectives
- Collects bodies of research paper links and classify the paper that is relevant to our specific domain or not. 
- Database Creation: Build a comprehensive dataset from scientific literature detailing molecules, their interactions with perovskites, and the outcomes (efficiency, stability, etc.).
- Molecular Representation: Use SMILES (Simplified Molecular Input Line Entry System) to represent molecules in a format suitable for machine learning models.
- Literature Mining: Automate data extraction from published research papers using NLP techniques (e.g., SciBERT, scraping tools).
<img src="images\pipeline.png" alt="pipeline" width="1000">


# Running the project
- To install the dependencies, run the following command from the root directory of the project: <code>pip install -r requirements.txt</code>
- To use GROBID (accessed in notebooks_for_checkpoint/xml_generator.ipynb), install [docker](https://docs.docker.com/engine/install/), 
then run the following command to download and run GROBID's image: <code>docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1</code>. This will initialize GROBID on http://localhost:8070.

## General Guidelines and Explanation of Our Work
All relevant files for this Quarter 2 is in the `q2_submission_notebooks` folder.

### 1: Generating a Relevant vs. Irrelevant Research Papers Database for Classification

**`00_crossref_scraping.ipynb`**  
   Scraping through urls of papers and their references to collect papers.

**`01_Scrapint_texts.ipynb`**  
   Ensure that two CSV files, `150_research_papers.csv` and `irrelevant_papers.csv`, are available in the `data` folder. This notebook accesses these research papers via URLs and extracts clean text data.  
   - **Output**: Upon completion, it outputs a `merged_label.csv` file in the `data` folder, which is used for classification testing.

**`02_TF-IDF_vectorizer_and_models.ipynb`**  
   This notebook performs five classification algorithms: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost  

   Each model undergoes hyperparameter tuning to find optimal settings. The results are saved as CSV files in the `data/model_results` folder, and visualizations show the performance before and after tuning.

**`03_scraping_and_conversion.ipynb`**  
This notebook gets the PDF file of articles and converts them into txt and xml files using Grobid.
   - **Output**: The files are saved in the `data/txts` and `data/xmls` folders respectively.

**`04_docling.ipynb`**  
Tested out docling to scan the paper from top to bottom to extract text and tables. It does well in getting clear table extraction but the text is very unorganized, therefore we will stick to Grobid text extraction.

### 2: Data Extracting Models

**`05_pretrained_model_extraction.ipynb`**  
   Using a pretrained model to extract data from text.
   - **Output**: data/finetuned_llama_output_1epoch.json

**`06_chatextract_openai.py`**  
   Attempted the implementation from the works of Maciej P. Polak and Dane Morgan. However, it will not be feasible since we require credit to run these extraction.
   
**`07_chunked_training_creation.ipynb`**  
Create text chunks so that model can extract data from a smaller portion of text.
   - **Output**: "data/chunked_example.csv",

### 3: Fine Tuning and Evaluation

**`08_finetuning_training.ipynb`**  
This notebook trains and prepare the model to be able to extract the data we need.

**`09_finetuning_test.ipynb`**  
This notebook tests performance on the chunked data.

**`10_evaluation_final.ipynb`**  
This notebook compares our text annotation with Extraction performed using precision score, recall score, and f1 score as its metric.




