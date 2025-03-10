# PSC-Passivator-Optimization

### Introduction
This project aims to optimize the discovery of small molecules that improve the stability of perovskite solar cells. By leveraging literature mining, graph-based molecular representations, and machine learning models, we seek to identify patterns that lead to successful molecules and generate deeper insights into perovskite solar cell performance. The current methods rely heavily on Edisonian experimentation, which is inefficient. Our goal is to automate and streamline this discovery process using data-driven techniques, focusing on the vast body of research already conducted in this field.
### 📂 Project Structure
```
PSC-Passivator-Optimization
├── data                                  # Collection of all data
│   ├── annotations/                      
│   ├── classification/                   
│   ├── extraction_eval/                  
│   ├── extraction_final/                 
│   ├── finetuning/                       
│   ├── model_results/                    
│   ├── performance_results/              
│   ├── prediction/                       
│   ├── prompts/                          
│   ├── rag_processing/                   
│   └── scraping_and_conversion/          
│
├── docs                                  # Documentation files
│   ├── index.html                        
│   ├── script.js                         
│   └── style.css                         
│
├── models                                # Model files and directories
│   ├── classifier_svm.pkl                
│   ├── classifier_xgb.pkl                
│   ├── DeepSeek-R1-PSC-Extractor-8B      
│   ├── DeepSeek-R1-PSC-Extractor-8B-8bit 
│   ├── DeepSeek-R1-PSC-Extractor-8B-8bit-Schema-2
│   ├── llama-3.2-3b-it-Perovskite-PaperExtractor
│   ├── Llama-PSC-Extractor-3B-16bit      
│   ├── LLama-PSC-Extractor-8B-8bit-Schema-2
│   └── scibert_psc_ner_model             
│
├── notebooks                             # Jupyter notebooks
│   ├── annotation_parsing_flattened.ipynb
│   ├── annotation_parsing_nested.ipynb   
│   ├── annotations_EDA.ipynb             
│   ├── chunked_training_creation_schema2.ipynb
│   ├── chunked_training_creation.ipynb   
│   ├── classifier_model_EDA.ipynb        
│   ├── docling.ipynb                     
│   ├── evaluation_final_finetuned.ipynb  
│   ├── evaluation_final_newschema.ipynb  
│   ├── evaluation_original.ipynb         
│   ├── extraction_vis.ipynb              
│   ├── prediction_vis.ipynb              
│   └── prediction2_vis.ipynb             
│
├── reports                               # Reports and figures
│   └── figures/                          
│
├── src                                   # Source code files
│   ├── __init__.py                       
│   ├── chunked_training_creation.py      
│   ├── classification.py                 
│   ├── crossref_dataset_generation.py    
│   ├── db_processing.py                  
│   ├── extraction_evaluation.py          
│   ├── extraction.py                     
│   ├── finetuning.py                     
│   ├── format_extraction.py              
│   ├── pdf_conversion.py                 
│   ├── prediction.py                     
│   ├── rag_filtering.py                  
│   └── scraping.py                       
│
├── requirements.txt                      # Python dependencies
└── run.py                                # Main execution script
```
Website: https://kanggunh.github.io/PSC-Passivator-Optimization/

### Objectives
- **Automate Data Extraction:** Implement a machine learning pipeline to mine relevant data from scientific literature.
- **Build a Comprehensive Database:** Structure data on perovskite compositions, passivating molecules, and experimental outcomes.
- **Molecular Representation:** Use SMILES to convert chemical information into a format suitable for modeling.
- **Develop Predictive Models:** Identify relationships between molecular features and performance.
- **Enhance Research Efficiency:** Replace manual experimentation with data-driven predictions to guide lab testing.

# Running the project
- To install the dependencies, run the following command from the root directory of the project: <code>pip install -r requirements.txt</code>
- To use GROBID (accessed in notebooks_for_checkpoint/xml_generator.ipynb), install [docker](https://docs.docker.com/engine/install/), 
then run the following command to download and run GROBID's image: <code>docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1</code>. This will initialize GROBID on http://localhost:8070.





