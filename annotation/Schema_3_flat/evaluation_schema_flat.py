### This file compares teamtat annotation with extraction performed on 3rd. flat schema
from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import numpy as np
import json
import os
import xml.etree.ElementTree as ET 
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

### Annotation Preparation
def str_toJson(string):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    # json_string = string.replace("None", "null")
    json_string = json.dumps(string)
    try:
        # Try to load the JSON string
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        # Catch JSONDecodeError if the string is not valid JSON
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return None
    
def convert_numeric(dictionary):
    ## Convert all numerical data into float for both
    if dictionary == None:
        return None
    numerical_key = ['time', 'efficiency_cont', 'efficiency_tret', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    exception_numeric = ['humidity', 'temperature']

    translation_table = str.maketrans('', '', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()')
    for key in dictionary.keys():
        if (key.startswith('test')) & (type(dictionary[key]) == dict):
            for entity in dictionary[key].keys():
                if entity in numerical_key:
                    # print(dictionary[key][entity])
                    if isinstance(dictionary[key][entity], str): 
                        substitute = re.sub(r'[^0-9.]', '', dictionary[key][entity][:4])
                        if len(substitute) != 0:
                            numerical_value = float(substitute)
                            dictionary[key][entity] = numerical_value
                        else:
                            dictionary[key][entity] = None
                elif entity in exception_numeric:
                    if isinstance(dictionary[key][entity], str): 
                        if "-" not in dictionary[key][entity]:
                            # print("regular_case",dictionary[key][entity])
                            substitute = re.sub(r'[^0-9.]', '', dictionary[key][entity][:4])
                            if len(substitute) != 0:
                                numerical_value = float(substitute)
                                dictionary[key][entity] = numerical_value
                            else:
                                dictionary[key][entity] = None
                        #     print(dictionary[key][entity])
        elif ('test_' in key) & (type(dictionary[key]) == dict):
            for entity in dictionary[key].keys():
                if entity in numerical_key:
                    # print(dictionary[key][entity])
                    if isinstance(dictionary[key][entity], str): 
                        substitute = re.sub(r'[^0-9.]', '', dictionary[key][entity][:4])
                        if len(substitute) != 0:
                            numerical_value = float(substitute)
                            dictionary[key][entity] = numerical_value
                        else:
                            dictionary[key][entity] = None
                elif entity in exception_numeric:
                    if isinstance(dictionary[key][entity], str): 
                        if "-" not in dictionary[key][entity]:
                            # print("regular_case",dictionary[key][entity])
                            substitute = re.sub(r'[^0-9.]', '', dictionary[key][entity][:4])
                            if len(substitute) != 0:
                                numerical_value = float(substitute)
                                dictionary[key][entity] = numerical_value
                            else:
                                dictionary[key][entity] = None
                        #     print(dictionary[key][entity])
        elif key in numerical_key:
            if isinstance(dictionary[key], str): 
                substitute = re.sub(r'[^0-9.]', '', dictionary[key][:4])
                if len(substitute) != 0:
                    numerical_value = float(substitute)
                    dictionary[key] = numerical_value
                else:
                    dictionary[key] = None
        elif key in exception_numeric:
            if isinstance(dictionary[key], str): 
                if "-" not in dictionary[key]:
                    # print("regular_case",dictionary[key][entity])
                    substitute = re.sub(r'[^0-9.]', '', dictionary[key][:4])
                    if len(substitute) != 0:
                        numerical_value = float(substitute)
                        dictionary[key] = numerical_value
                    else:
                        dictionary[key] = None
    return dictionary

def convert_efficiency(dictionary):
    entity_decimal = ['efficiency_cont','efficiency_tret']
    for key in dictionary.keys():
        if (key.startswith('test')) & (type(dictionary[key]) == dict):
            for entity in dictionary[key].keys():
                if (entity in entity_decimal) and (dictionary[key][entity] != None):
                    if dictionary[key][entity] == dictionary[key][entity] > 1:
                        dictionary[key][entity] = dictionary[key][entity] / 100
    return dictionary

##Loading in annotation
with open('data/annotations_flattened.json', 'r') as f:
    json_data = json.load(f)

flattened_format = []
for key in json_data:
    papers = json_data[key]
    if papers is None:
        flattened_format.append({ "paper_id": key, "output": None })
        continue
    if len(papers.keys()) == 0:
        print(key)
    for passivator in papers:
        paper_data = papers[passivator]
        paper_keys = paper_data.keys()
        test_keys = [key for key in paper_keys if "test" in key]
        for test_key in test_keys:
            flattened_paper = {k: v for k, v in paper_data.items() if k not in test_keys}
            flattened_paper.update(paper_data[test_key])
            flattened_format.append({ "paper_id": int(key), "output": flattened_paper })

annotation_df = pd.DataFrame(flattened_format)
annotation_df.columns = ['paper_num', 'output']
annotation_df["paper_num"] = annotation_df["paper_num"].astype(int)
annotation_df = annotation_df.sort_values(by = 'paper_num')
annotation_df = annotation_df[annotation_df["output"].isna() == False]
annotation_df["paper_num"] = pd.to_numeric(annotation_df["paper_num"])

## Getting Extraction JSON

def convert_efficiency_key(dict):
    for key, item in dict.items():
        if 'test' in key:
            if 'retained_proportion_cont' in dict[key]:
                dict[key]['efficiency_cont'] = dict[key].pop('retained_proportion_cont')
            if 'retained_proportion_tret' in dict[key]:
                dict[key]['efficiency_tret'] = dict[key].pop('retained_proportion_tret')
    return dict

## extraction performed by basemodel
with open("data/deepseek_base_flat.json", 'r') as f:
    extraction = json.load(f)

extraction_base = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_base['paper_num'] = pd.to_numeric(extraction_base['paper_num'])
extraction_base = extraction_base.sort_values('paper_num')
extraction_base['output'] = extraction_base['output'].apply(convert_numeric)
extraction_base['output'] = extraction_base['output'].apply(convert_efficiency)




