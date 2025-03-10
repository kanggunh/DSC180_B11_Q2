# PSC-Passivator-Optimization

### Introduction
This project aims to optimize the discovery of small molecules that improve the stability of perovskite solar cells. By leveraging literature mining, graph-based molecular representations, and machine learning models, we seek to identify patterns that lead to successful molecules and generate deeper insights into perovskite solar cell performance. The current methods rely heavily on Edisonian experimentation, which is inefficient. Our goal is to automate and streamline this discovery process using data-driven techniques, focusing on the vast body of research already conducted in this field.

**Visit the project website:** [PSC-Passivator-Optimization](https://kanggunh.github.io/PSC-Passivator-Optimization/)

---
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
---

## Running the project
**1. Install the dependencies:** run the following command from the root directory of the project: 
```
pip install -r requirements.txt
```
**2. Set Up Environment Variables:** Create a .env file in the root directory and add the following:
```
HF_TOKEN=your_huggingface_token
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
```
**3. Set Up GROBID for PDF Conversion**
- install [docker](https://docs.docker.com/engine/install/),
- Run the following command to download and run GROBID's image:
```
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1
```
- This will initialize GROBID on http://localhost:8070.
## Using run.py

**1. Run the Full Pipeline:** <code>python run.py all</code>

**2. Run Specific Stages:**
- <code>python run.py scraping_and_conversion</code>
- <code>python run.py classification</code>
- <code>python run.py finetuning</code>
- <code>python run.py extraction_evaluation</code>
- <code>python run.py extraction</code>
- <code>python run.py prediction</code>






