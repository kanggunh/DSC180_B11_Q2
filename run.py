# import subprocess
# import os

# def run_script(script_path):
#     """Run a Python script."""
#     print(f"Running script: {script_path}")
#     subprocess.run(['python', script_path], check=True)

# def run_notebook(notebook_path):
#     """Run a Jupyter notebook."""
#     print(f"Running notebook: {notebook_path}")
#     subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path], check=True)

# def main():
#     # Step 1: Generating a Relevant vs. Irrelevant Research Papers Database for Classification
    
#     # Step 1.1: Run the first script
#     run_script('notebooks_for_checkpoint/1_extract_link_badpaper.py')

#     # Step 1.2: Run the second notebook
#     run_notebook('notebooks_for_checkpoint/2_Scrapint_texts.ipynb')

#     # Step 2: Building and Evaluating Classification Models
    
#     # Step 2.1: Run the third notebook
#     run_notebook('notebooks_for_checkpoint/3_TF-IDF_vectorizer_and_models.ipynb')
    
#     # Step 2.2: Run the fourth notebook
#     run_notebook('notebooks_for_checkpoint/4_flan_model.ipynb')

#     # Step 2.3: Run the fifth notebook
#     run_notebook('notebooks_for_checkpoint/5_sciBERT.ipynb')

#     # Step 2.4: Run the sixth notebook
#     run_notebook('notebooks_for_checkpoint/6_model_analysis.ipynb')

#     # Additional Tasks for Data Extraction
    
#     # Step 2.5: Run the seventh notebook
#     run_notebook('notebooks_for_checkpoint/7_database_searcher.ipynb')

#     # Step 2.6: Run the eighth notebook
#     run_notebook('notebooks_for_checkpoint/8_xml_generator.ipynb')

#     print("All tasks completed successfully.")

# if __name__ == "__main__":
#     main()
