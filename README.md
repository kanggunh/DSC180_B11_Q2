# Perovskite Solar Cells: Literature Mining Project

### Folder Structure
```
DSC180_B11
└───data
│    │   biocs/                                
│    │   model_results/                        
│    │   txts/                               
│    │   xmls/                                 
│    │   150_research_papers.csv           # Dataset of 150 research papers
│    │   bioc_parsed.csv                       
│    │   good_paper_links.csv                  
│    │   irrelevant_papers.csv                 
│    │   merged_label.csv                  # labeled dataset for classification 
│    │   model_performance_results.csv         
│    │   Perovskite_database_content_all_data.csv 
│    │   text_bad_paper.csv                   
│    │   text_good_paper.csv                   
│    └───training_data.csv                 # Dataset for training the models
│
└───images
│    │   classification_compare.png        # Model comparison visualization
│    │   pipeline_flowchart.png            # Workflow diagram
│    └───q2_timeline.png                   # Project timeline visualization
│
└───models
│    │   llama-3.2-3b-it-Perovskite-PaperExtractor/ 
│    └───scibert_psc_ner_model/                
│
└───q1_submission_notebooks
│    │   01_extract_link_badpaper.py           
│    │   02_Scrapint_texts.ipynb              
│    │   03_TF-IDF_vectorizer_and_models.ipynb 
│    │   04_flan_model.ipynb                   
│    │   05_sciBERT.ipynb                      
│    │   06_scraping_and_conversion.ipynb      
│    │   07_llama_training.ipynb               
│    │   08_finetuned_llama_extraction.ipynb   
│    │   09_evaluation.ipynb                   
│    │   10_scibert_training.ipynb             
│    └───11_database_searcher.ipynb            
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
<img src="images\pipeline_flowchart.png" alt="pipeline" width="1000">


# Running the project
- To install the dependencies, run the following command from the root directory of the project: <code>pip install -r requirements.txt</code>
- To use GROBID (accessed in notebooks_for_checkpoint/xml_generator.ipynb), install [docker](https://docs.docker.com/engine/install/), 
then run the following command to download and run GROBID's image: <code>docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1</code>. This will initialize GROBID on http://localhost:8070.

## General Guidelines and Explanation of Our Work
All relevant files for this Quarter 1 Submission is in the `q1_submission_notebooks` folder.

### 1: Generating a Relevant vs. Irrelevant Research Papers Database for Classification

**`01_extract_link_badpaper.py`**  
   This script takes two URLs of irrelevant research papers, performs literature mining, and scrapes all references from these papers. These references form a training set representing irrelevant papers.  
   - **Output**: After successful execution, it generates an `irrelevant_papers.csv` file in the `data` folder.

**`02_Scrapint_texts.ipynb`**  
   Ensure that two CSV files, `150_research_papers.csv` and `irrelevant_papers.csv`, are available in the `data` folder. This notebook accesses these research papers via URLs and extracts clean text data.  
   - **Output**: Upon completion, it outputs a `merged_label.csv` file in the `data` folder, which is used for classification testing.

### 2: Building and Evaluating Classification Models

**`03_TF-IDF_vectorizer_and_models.ipynb`**  
   This notebook performs five classification algorithms:
   - Logistic Regression
   - Naive Bayes
   - SVM
   - Random Forest
   - XGBoost  

   Each model undergoes hyperparameter tuning to find optimal settings. The results are saved as CSV files in the `data/model_results` folder, and visualizations show the performance before and after tuning.

**`04_flan_model.ipynb`**  
   This notebook performs classification using Google’s Fine-tuned Language Net model.  
   - **Output**: The performance results are saved as a CSV file in the `data/model_results` folder.

**`05_sciBERT.ipynb`**  
   This notebook performs classification using the SciBERT model.  
   - **Output**: The performance results are saved as a CSV file in the `data/model_results` folder.

**`06_model_analysis.ipynb`**  
   This notebook aggregates the results from all models and compares the Accuracy, Recall, and Balanced Error Rate (BER) metrics for each model. Visualizations illustrate these comparisons, showing Random Forest with the highest Accuracy and BER.
   - **Visualization**:    
     <img src="images\classification_compare.png" alt="Comparing performance metric for ALL classification perfomed " width="400">

### 3: Scraping and Converting Research Articles

**`07_scraping_and_conversion.ipynb`**  
   This notebook gets the PDF file of articles and converts them into txt and xml files.  
   - **Output**: The files are saved in the `data/txts` and `data/xmls` folders respectively.

### 4: Training and Extracting Data
**`08_llama_training.ipynb`**  
   This notebook trains a Llama model using the `training_data.csv`.
   - **Output**: It saves the trained model in `models/llama-3.2-3b-it-Perovskite-PaperExtractor` folder.

**`09_finetuned_llama_extraction.ipynb`**  
   Run the extraction process using the model from `models/llama-3.2-3b-it-Perovskite-PaperExtractor` folder.
   - **Output**: The output of the extraction is then saved to `data/finetuned_llama_output.json` folder. 

**`10_evaluation.ipynb`**  
   This notebook evaluates the extraction using F1 Score, MacroF1 Score, Precision, and Recall.

### Additional Tools and Notebooks

**`11_scibert_training.ipynb`**  
   This notebook trains a sciBERT model for data extraction.  
   - **Output**: It saves the trained model in `data/scibert_psc_ner_model` folder.

**`12_database_searcher.ipynb`**  
   This notebook includes a function that retrieves key variables (e.g., efficiency, condition, PCE) from a given research paper's DOI using the Perovskite Database. It is intended for use in the extraction phase of our project.

