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




