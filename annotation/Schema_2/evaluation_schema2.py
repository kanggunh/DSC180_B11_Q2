## This file compares teamtat annotation with extraction performed on 2nd, semi nested Schema
from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import numpy as np
import json
import os
import xml.etree.ElementTree as ET 
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

## Numerical list o relevent and irrelevent numbers in 0-140 papers
irrelevent_papers = [26,28,29,30,32,33,34,35,38,43,44,45,52,54,55,56,57,58,63,66,68,69,70,78,80,83,84,86,87,88,89,90,91,92,93,94,98,100,101,102,103,104,105,106,108,109,110,111,112,115,116,117,119,121,125,128,129,130,132,134,136,138,139, 140]
relevent_bad = [1, 18, 20, 25, 27, 41, 51, 61, 67, 71, 76, 135, 141, 145]

relevent_good = [i for i in range(0, 150) if i not in irrelevent_papers and i not in relevent_bad]

#### File preparation to ensure that both annotation and extraction are in correct JSON
def str_toJson(string):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    json_string = string.replace("None", "null")

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
    
def convert_numeric_outside(dictionary):
    key_list = ['control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    for key in key_list:
        if dictionary[key] != None:
            dictionary[key] = float(dictionary[key])
    return dictionary

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

def filter_output(dictionary):
    ## I want to drop row where passivation is none. 
    if dictionary['passivating_molecule'] == None:
        return True
    return False

##Loading into TeamTat Annotation
with open('data/annotations_flattened.json', 'r') as f:
    json_data = json.load(f)

flattened_format = []
for key in json_data:
    papers = json_data[key]
    if papers is None:
        flattened_format.append({ "paper_id": key, "output": None })
        continue
    for passivator in papers:
        paper_data = papers[passivator]
        flattened_format.append({ "paper_id": key, "output": paper_data })

annotation_df = pd.DataFrame(flattened_format)
annotation_df.columns = ['paper_num', 'output']
annotation_df["paper_num"] = annotation_df["paper_num"].astype(int)
annotation_df = annotation_df.sort_values(by = 'paper_num')

## Get the annotation_df with only relevent good papers. 
annotation_df = annotation_df[annotation_df['paper_num'].isin(relevent_good)]
## Get all the relevent present papers for now
annotation_df = annotation_df[annotation_df['output'].notnull()]

#There is rows where the duplicated paper num has a dictionary that has passivating missing. We will drop these rows
annotation_filter = annotation_df['output'].apply(filter_output)
annotation_df['filter'] = annotation_filter
annotation_df = annotation_df[annotation_df['filter'] == False][['paper_num', 'output']]
annotation_df['output'] = annotation_df['output'].apply(convert_numeric)
annotation_df['output'] = annotation_df['output'].apply(convert_numeric_outside)


## Loading JSON extraction
def convert_efficiency_key(dict):
    if dict == None:
        data_schema = {
            'perovskite_composition': None,
            'electron_transport_layer': None,
            'hole_transport_layer': None,
            'structure_pin_nip': None,
            'passivating_molecule': None,
            'control_pce': None,
            'treated_pce': None,
            'control_voc': None,
            'treated_voc': None,
            'test_1': {
                'stability_type': None,
                'humidity': None,
                'temperature': None,
                'time': None,
                'efficiency_tret': None,
                'efficiency_cont': None
            }
        }
        return data_schema
    for key, item in dict.items():
        if 'test' in key:
            if dict[key] == None:
                return dict
            if 'retained_proportion_cont' in dict[key]:
                dict[key]['efficiency_cont'] = dict[key].pop('retained_proportion_cont')
            if 'retained_proportion_tret' in dict[key]:
                dict[key]['efficiency_tret'] = dict[key].pop('retained_proportion_tret')

            if 'retained_percentage_cont' in dict[key]:
                dict[key]['efficiency_cont'] = dict[key].pop('retained_percentage_cont')
            if 'retained_percentage_tret' in dict[key]:
                dict[key]['efficiency_tret'] = dict[key].pop('retained_percentage_tret')

            if 'test_name' in dict[key]:
                dict[key]['stability_type'] = dict[key].pop('test_name')
    return dict

def convert_structure_key(dict):
    # print(dict)
    found = 0
    found_2 = 0
    for key, item in dict.items():
        if key == 'pin_nip_structure':
            value = dict[key]
            found = 1
        if key == "treated_pec":
            value_2 = dict[key]
            found_2 = 1
    if found == 1:
        dict['structure_pin_nip'] = dict.pop('pin_nip_structure')
    if found_2 == 1:
        dict['treated_pce'] = dict.pop('treated_pec')
    
    return dict

def force_fix(json_string):
    '''
    {"perovskite_composition": "Cs0.05(FA0.98MA0.02)0.95Pb(I0.98Br0.02)3", "electron_transport_layer": "6,6"-phenyl-C61-butyric acid methyl ester", "pin_nip_structure": "NIP", "hole_transport_layer": "2-(3,6-dimethoxy-9H-carbazol-9-yl)ethyl phosphonic acid", "test_1": {"test_name": "ISOS-L", "temperature": "25", "time": "1000", "humidity": "50-60", "passivating_molecule": "β-poly(1,1-difluoroethylene)", "control_pce": "22.3", "treated_pce": "24.6", "control_voc": "1.13", "treated_voc": "1.18"}}
    '''
    ## convert {" pattern to ZS
    json_conv = re.sub(r'{\"',"ZS", json_string)
    ## convert ": to ZT
    json_conv = re.sub(r'\":', "ZT", json_conv)
    ## Convert _" to ZP
    json_conv = re.sub(r' \"', "ZP", json_conv)
    ## Convert ", to ZV
    json_conv = re.sub(r'\",Z', "ZVZ", json_conv)
    ## Convert "} to ZQ
    json_conv = re.sub(r'\"}', "ZQ", json_conv)

    ##Perform real conversion interested
    json_conv = json_conv.replace('"', "'")

    #Revert Everything 

    ## convert ZS to {" 
    json_conv = re.sub("ZS", r'{"', json_conv)
    ## convert ": to ZT
    json_conv = re.sub("ZT", r'":', json_conv)
    ## Convert _" to ZP
    json_conv = re.sub("ZP", r' "', json_conv)
    ## Convert ", to ZV
    json_conv = re.sub("ZV", r'",', json_conv)
    ## Convert "} to ZQ
    json_conv = re.sub("ZQ", r'"}', json_conv)

    return json_conv

def escape_internal_quotes(json_string):
    # This regex finds values within quotes and fixes internal unescaped quotes
    return re.sub(r'":\s*"([^"]*?)"', lambda m: '": "' + m.group(1).replace('"', '\\"') + '"', json_string)

def str_toJson_8bit(string):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    if string == None:
        return None
    
    json_string = string 
    json_string = string.replace("None", "null")
    # json_string = json_string.replace(r'\"', '"')
    json_string = json_string.replace("'", '"')
    json_string = escape_internal_quotes(json_string)
    json_string = force_fix(json_string)

    try:
        # Try to load the JSON string
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        # Catch JSONDecodeError if the string is not valid JSON
        print(f"Error decoding JSON: {e}")
        print(json_string)
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return None

## extraction performed by basemodel
with open("annotation\Schema_2\data\deepseek_base_updateschema.json", 'r') as f:
    extraction = json.load(f)

extraction_base = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_base['paper_num'] = pd.to_numeric(extraction_base['paper_num'])
extraction_base = extraction_base.sort_values('paper_num')
extraction_base['output'] = extraction_base['output'].apply(convert_numeric)
extraction_base = extraction_base[extraction_base['paper_num'].isin(relevent_good)]

## extraction performed by finetuned deepseep
with open("annotation\Schema_2\data\deepseek_8bit_finetuned.json", 'r') as f:
    extraction = json.load(f)

extraction_deepseek = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_deepseek['paper_num'] = pd.to_numeric(extraction_deepseek['paper_num'])
extraction_deepseek = extraction_deepseek.sort_values('paper_num')
extraction_deepseek['output'] = extraction_deepseek['output'].apply(convert_efficiency_key)
extraction_deepseek['output'] = extraction_deepseek['output'].apply(convert_structure_key)
extraction_deepseek['output'] = extraction_deepseek['output'].apply(convert_numeric)
extraction_deepseek = extraction_deepseek[extraction_deepseek['paper_num'].isin(relevent_good)]

## extraction performed by llama 
with open("annotation\Schema_2\data\llama_8bit_finetuned.json", 'r') as f:
    extraction = json.load(f)

extraction_llama = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_llama['paper_num'] = pd.to_numeric(extraction_llama['paper_num'])
extraction_llama = extraction_llama.sort_values('paper_num')
extraction_llama['output'] = extraction_llama['output'].apply(convert_efficiency_key)
extraction_llama['output'] = extraction_llama['output'].apply(convert_structure_key)
extraction_llama['output'] = extraction_llama['output'].apply(convert_numeric)


##Merging dataframe

def get_passivation(dict):
    ## Get the passivation from given dictionary
    if "passivating_molecule" not in dict.keys():
        return None
    if type(dict['passivating_molecule']) == list:
        return dict['passivating_molecule'][0]
    return dict['passivating_molecule']

def compare_passivation(tuple):
    # print(tuple[0])
    # print(tuple[1])
    if tuple[1] == None:
        return 0
    similarity = SequenceMatcher(None, tuple[0].lower(), tuple[1].lower()).ratio()
    return similarity

evaluate_df_base = annotation_df.merge(extraction_base, left_on='paper_num', right_on='paper_num')[["paper_num", "output_x",'output_y']]
evaluate_df_base.columns = ['paper_num', 'annotation', 'extracted']
annotation_passivation = evaluate_df_base['annotation'].apply(get_passivation)
extraction_passivation = evaluate_df_base['extracted'].apply(get_passivation)
# Combine them into a tuple
combined_tuples = list(zip(annotation_passivation, extraction_passivation))
tuple_series = pd.Series(combined_tuples)
evaluate_df_base['passivations'] = tuple_series
similarity = evaluate_df_base['passivations'].apply(compare_passivation)
evaluate_df_base['similarity'] = similarity
## Get the row where the similarity was maximum for groupby paper_num
evaluate_df_base = evaluate_df_base.loc[evaluate_df_base.groupby("paper_num")["similarity"].idxmax()]


evaluate_df_deepseek = annotation_df.merge(extraction_deepseek, left_on='paper_num', right_on='paper_num')[["paper_num", "output_x",'output_y']]
evaluate_df_deepseek.columns = ['paper_num', 'annotation', 'extracted']
annotation_passivation = evaluate_df_deepseek['annotation'].apply(get_passivation)
extraction_passivation = evaluate_df_deepseek['extracted'].apply(get_passivation)
# Combine them into a tuple
combined_tuples = list(zip(annotation_passivation, extraction_passivation))
tuple_series = pd.Series(combined_tuples)
evaluate_df_deepseek['passivations'] = tuple_series
similarity = evaluate_df_deepseek['passivations'].apply(compare_passivation)
evaluate_df_deepseek['similarity'] = similarity
## Get the row where the similarity was maximum for groupby paper_num
evaluate_df_deepseek = evaluate_df_deepseek.loc[evaluate_df_deepseek.groupby("paper_num")["similarity"].idxmax()]


evaluate_df_llama = annotation_df.merge(extraction_llama, left_on='paper_num', right_on='paper_num')[["paper_num", "output_x",'output_y']]
evaluate_df_llama.columns = ['paper_num', 'annotation', 'extracted']
annotation_passivation = evaluate_df_llama['annotation'].apply(get_passivation)
extraction_passivation = evaluate_df_llama['extracted'].apply(get_passivation)
# Combine them into a tuple
combined_tuples = list(zip(annotation_passivation, extraction_passivation))
tuple_series = pd.Series(combined_tuples)
evaluate_df_llama['passivations'] = tuple_series
similarity = evaluate_df_llama['passivations'].apply(compare_passivation)
evaluate_df_llama['similarity'] = similarity
## Get the row where the similarity was maximum for groupby paper_num
evaluate_df_llama = evaluate_df_llama.loc[evaluate_df_llama.groupby("paper_num")["similarity"].idxmax()]


## Evaluations

def tests_comparison(stability_annotated, label_dict, stability_extracted, extract_dict):
    # print(stability_annotated, label_dict, stability_extracted, extract_dict)
    stability_entity_annotated = ['stability_type', 'temperature', 'time', 'humidity', 'efficiency_cont', 'efficiency_tret']
    # stability_entity_extracted = ['test_name', 'passivating_molecule', 'temperature', 'time', 'humidity', 'control_efficiency', 'treatment_efficiency', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    
    # print(f"stability_annotated{stability_annotated}")
    # print(f"label_dict{label_dict}")
    # print(f"stability_extracted{stability_extracted}")
    # print(f"extract_dict{extract_dict}")
    compared_metric = []
    numeric_data_annotated = []
    numeric_data_extracted = []
    for entity_i in range(len(stability_entity_annotated)):
        if entity_i < 1:
            # print(stability_annotated[entity_i])
            if stability_entity_annotated[entity_i] not in extract_dict.keys():
                extract_dict[stability_entity_annotated[entity_i]] = None

            if (label_dict[stability_entity_annotated[entity_i]] == None) | (extract_dict[stability_entity_annotated[entity_i]] == None):
                compared_metric.append(None)
            else:
                ##Text entity, perform Sequence Matcher 
                compared = SequenceMatcher(None, label_dict[stability_entity_annotated[entity_i]], extract_dict[stability_entity_annotated[entity_i]]).ratio()
                # print(compared)
                if entity_i == 0:
                    if compared > 0.9:
                        compared_metric.append(1)
                    else:
                        compared_metric.append(0)
                else:
                    compared_metric.append(compared)
        else:
            if stability_entity_annotated[entity_i] not in extract_dict.keys():
                extract_dict[stability_entity_annotated[entity_i]] = 0
            elif extract_dict[stability_entity_annotated[entity_i]] == None:
                extract_dict[stability_entity_annotated[entity_i]] = 0

            if stability_entity_annotated[entity_i] not in label_dict.keys():
                label_dict[stability_entity_annotated[entity_i]] = 0
            elif label_dict[stability_entity_annotated[entity_i]] == None:
                label_dict[stability_entity_annotated[entity_i]] = 0

                
            numeric_data_annotated.append(label_dict[stability_entity_annotated[entity_i]])
            numeric_data_extracted.append(extract_dict[stability_entity_annotated[entity_i]])

    if isinstance(numeric_data_extracted[0], list):
        ##There was one column with two temperature recorded as a list (probably thermal cycling)
        numeric_data_extracted[0] = numeric_data_extracted[0][1]

    # print(numeric_data_annotated, numeric_data_extracted)

    numeric_annotated_clean = []
    numeric_extracted_clean = []
    ##Clean the numeric data to skip any strings
    for i in range(len(numeric_data_annotated)):
        if (type(numeric_data_annotated[i]) == str) | (type(numeric_data_extracted[i]) == str):
            continue
        else:
            numeric_annotated_clean.append(numeric_data_annotated[i])
            numeric_extracted_clean.append(numeric_data_extracted[i])

    cos_sim = cosine_similarity([numeric_annotated_clean], [numeric_extracted_clean])
    compared_metric.append(cos_sim[0][0])
    
    return compared_metric   

def entity_comparison(entity, label, extracted_dict, text_similarity_threshold = 0.75, numerical_tolerance = 0.027):
    '''
    The tolarance of 2.7% was what was reasonable looking at the absolute difference
    treated_voc 1.18, 1.149, absolute difference 0.026271186440677895

    The text similarity were set to 75% due to the structure example
    FP, NIP, n-i-p, 0.75
    This should be positive
    
    '''
    text_entity = ['stability_type']
    numerical_entity = ['time', 'efficiency_cont', 'efficiency_tret']
    numerical_exception = ['temperature', 'humidity']

    if entity in text_entity:
        # key_to_check = "test_name" if entity == "stability_type" else entity

        # If the key is missing in the extracted annotation, return False Negative
        if (label[entity]!=None) & (extracted_dict[entity]==None):
            # print(f"FN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
            return "FN"
        elif (label[entity]==None) & (extracted_dict[entity]!=None):
            # print(f"TN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
            return "TN"

        label_data = label.get(entity, "")
        extract_data = extracted_dict.get(entity, "")

        # Convert lists to strings if necessary
        if isinstance(label_data, list):
            label_data = " ".join(map(str, label_data))  # Convert list to string
        if isinstance(extract_data, list):
            extract_data = " ".join(map(str, extract_data))  # Convert list to string

        # Ensure values are strings
        if not isinstance(label_data, str) or not isinstance(extract_data, str):
            # print(f"FP, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
            return "FP"  # If data is still not a string, return False Positive

        # Compute similarity score
        similarity = SequenceMatcher(None, label_data.lower(), extract_data.lower()).ratio()

        if similarity > text_similarity_threshold:
            # print(f"TP,{entity} {label_data}, {extract_data}")
            return 'TP'
        else:
            # print(f"FP,{entity} {label_data}, {extract_data}, {similarity}")
            return "FP"
    elif entity in numerical_entity:
        # key_to_check = "control_efficiency" if entity == "efficiency_cont" else ("treatment_efficiency" if entity == "efficiency_tret" else entity)
        # if (entity == 'efficiency_cont') | (entity == 'efficiency_tret'):
            # print(f"annotated{label[entity]}")
            # print(f"extracted{extracted_dict[entity]}")
        if extracted_dict[entity] == None:
            extracted_dict[entity] = 0

        # If the key is missing in the extracted annotation, return False Negative
        if (label[entity]!=0) & ((extracted_dict[entity]==0) | (entity not in extracted_dict.keys())):
            # print(f"FN, {label_annotation[id]}, {extraction_annotation[entity]}")
            return "FN"
        elif (label[entity]==0) & (extracted_dict[entity]!=0):
            # print(f"TN, {label_annotation[id]}, {extraction_annotation[entity]}")
            return "TN"
        elif (label[entity]==0) & (extracted_dict[entity]==0):
            # print(f"TN, {label_annotation[id]}, {extraction_annotation[entity]}")
            return "TN"


        if isinstance(extracted_dict[entity], list):
            ##There was one column with two temperature recorded as a list (probably thermal cycling)
            extracted_dict[entity] = extracted_dict[entity][1]
        

        # print(label[entity], extracted_dict[entity])
        # Apply numerical tolerance check
        if (abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )<= numerical_tolerance:

            # print(f"Numerical differences matched: {entity} {label[entity]}, {extracted_dict[entity]}, absolute difference {(abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )}")
            return "TP"  # True Positive: Correct numerical extraction
        else:

            # print(f"Numerical differences no match: {entity}, {label[entity]}, {extracted_dict[entity]}, absolute difference {(abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )}")
            return "FP"  # False Positive: Incorrect numerical extraction    
    else: 
        if isinstance(label[entity], (float, int)):
            if extracted_dict[entity] == None:
                extracted_dict[entity] = 0

            # If the key is missing in the extracted annotation, return False Negative
            if (label[entity]!=0) & ((extracted_dict[entity]==0) | (entity not in extracted_dict.keys())):
                # print(f"FN, {label_annotation[id]}, {extraction_annotation[entity]}")
                return "FN"
            elif (label[entity]==0) & (extracted_dict[entity]!=0):
                # print(f"TN, {label_annotation[id]}, {extraction_annotation[entity]}")
                return "TN"
            elif (label[entity]==0) & (extracted_dict[entity]==0):
                # print(f"TN, {label_annotation[id]}, {extraction_annotation[entity]}")
                return "TN"


            if isinstance(extracted_dict[entity], list):
                ##There was one column with two temperature recorded as a list (probably thermal cycling)
                extracted_dict[entity] = extracted_dict[entity][1]

            # Apply numerical tolerance check
            if (abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )<= numerical_tolerance:

                # print(f"Numerical differences matched: {entity} {label[entity]}, {extracted_dict[entity]}, absolute difference {(abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )}")
                return "TP"  # True Positive: Correct numerical extraction
            else:

                # print(f"Numerical differences no match: {entity}, {label[entity]}, {extracted_dict[entity]}, absolute difference {(abs(label[entity] - extracted_dict[entity])) / (abs(label[entity]) )}")
                return "FP"  # False Positive: Incorrect numerical extraction    
        else:
            # print(label[entity], type(label[entity]))
            if extracted_dict[entity] == None:
                extracted_dict[entity] = 0
            
            if ((extracted_dict[entity]==0) | (entity not in extracted_dict.keys())):
                # print(f"FN, {label_annotation[id]}, {extraction_annotation[entity]}")
                return "FN"

            if isinstance(extracted_dict[entity], list):
                ##There was one column with two temperature recorded as a list (probably thermal cycling)
                extracted_dict[entity] = extracted_dict[entity][1]
            
            if isinstance(extracted_dict[entity], str):
                ##Label is str, extraction is str, so perform text similarity
                similarity = SequenceMatcher(None, label[entity].lower(), extracted_dict[entity].lower()).ratio()
                if similarity > text_similarity_threshold:
                    # print(f"TP, {label_data}, {extract_data}, {similarity}")
                    return 'TP'
                else:
                    # print(f"FP, {label_data}, {extract_data}, {similarity}")
                    return "FP"
            else:
                if "+" in label[entity]:
                    # print(label[entity].split("+-"))
                    value = float(label[entity].split("+-")[0])
                    margin_error = float(label[entity].split("+-")[1])
                    range = (value-margin_error, value+margin_error)
                    if (range[0]<= extracted_dict[entity]) & (extracted_dict[entity]<=range[1]):
                        # print(f"TP, {label_data}, {extract_data}, {similarity}")
                        return 'TP'
                    else:
                        # print(f"FP, {label_data}, {extract_data}, {similarity}")
                        return "FP"
                else:
                    lower = float(label[entity].split("-")[0])
                    upper = float(label[entity].split("-")[1])
                    if (lower<= extracted_dict[entity]) & (extracted_dict[entity]<=upper):
                        # print(f"TP, {label_data}, {extract_data}, {similarity}")
                        return 'TP'
                    else:
                        # print(f"FP, {label_data}, {extract_data}, {similarity}")
                        return "FP"

def safe_division(numerator, denominator):
    """Returns division result, or 0 if the denominator is zero."""
    return numerator / denominator if denominator != 0 else 0

def text_comparison(id, label_annotation, extraction_annotation, text_similarity_threshold=0.8):
    """Compares text values using string similarity matching.
    - THE 4 basic variable that is to compare is PEROVSKITE COMPOSITION, ETL, HTL, STRUCTURE
    """

    # Handle special case for structure_pin_nip
    # key_to_check = "pin_nip_structure" if id == "structure_pin_nip" else id
    # print(label_annotation)
    # print(extraction_annotation)

    # If the key is missing in the extracted annotation, return False Negative
    if (label_annotation[id]!=None) & (extraction_annotation[id]==None):
        # print(f"FN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
        return "FN"
    elif (label_annotation[id]==None) & (extraction_annotation[id]!=None):
        # print(f"TN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
        return "TN"

    label_data = label_annotation.get(id, "")
    if id == 'electron_transport_layer' and label_data == "buckminsterfullerene":
        label_data = 'C60'
    extract_data = extraction_annotation.get(id, "")

    # Convert lists to strings if necessary
    if isinstance(label_data, list):
        label_data = " ".join(map(str, label_data))  # Convert list to string
    if isinstance(extract_data, list):
        extract_data = " ".join(map(str, extract_data))  # Convert list to string

    # Ensure values are strings
    if not isinstance(label_data, str) or not isinstance(extract_data, str):
        # print(f"FP, {label_annotation[id]}, {extraction_annotation[id]}")
        return "FP"  # If data is still not a string, return False Positive

    # Compute similarity score
    similarity = SequenceMatcher(None, label_data.lower(), extract_data.lower()).ratio()

    if similarity > text_similarity_threshold:
        # print(f"TP, {label_data}, {extract_data}, {similarity}")
        return 'TP'
    else:
        # print(f"FP, {label_data}, {extract_data}, {similarity}")
        return "FP"

def numeric_comoparison(id, label_value, extracted_value, numerical_tolerance = 0.027):
    # print(id)
    # print(f"label value: {label_value[id]}, {type(label_value[id])}")
    # print(extracted_value)
    # print(f"extract value: {extracted_value[id]}, {type(extracted_value[id])}")

    if (label_value[id]!=None) & (extracted_value[id]==None):
        # print(f"FN, {label_value[id]}, {extracted_value[id]}")
        return "FN"
    elif (label_value[id]==None) & (extracted_value[id]!=None):
        # print(f"TN, {label_value[id]}, {extracted_value[id]}")
        return "TN"
    elif (label_value[id]==None) & (extracted_value[id]==None):
        ##Anotation failed to extract and extraction didn't extract. This is TP
        # print(f"TP, {label_value[id]}, {extracted_value[id]}")
        return "TP"
    # Apply numerical tolerance check
    elif (abs(label_value[id] - extracted_value[id])) / (abs(label_value[id]) )<= numerical_tolerance:

        # print(f"Numerical differences matched: {id} {label_value[id]}, {extracted_value[id]}, absolute difference {(abs(label_value[id] - extracted_value[id])) / (abs(label_value[id]) )}")
        return "TP"  # True Positive: Correct numerical extraction
    else:
        # print(f"Numerical differences no match: {id}, {label_value[id]}, {extracted_value[id]}, absolute difference {(abs(label_value[id] - extracted_value[id])) / (abs(label_value[id]) )}")
        return "FP"  # False Positive: Incorrect numerical extraction    

def compare_json(df):
    """
    Compare labeled and extracted JSON data for correctness.

    TP: Correct value extracted by LLM.
    FN: LLM didn't extract this variable.
    FP: LLM extracted a value, but it was incorrect.
    TN: LLM halucinated and returned value that was not extracted
    """
    
    outside_variables = ['perovskite_composition', 'electron_transport_layer', 'hole_transport_layer', 'structure_pin_nip', "passivating_molecule", 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    outside_text = ['perovskite_composition', 'electron_transport_layer', 'hole_transport_layer', 'structure_pin_nip', "passivating_molecule"]
    
    stability_entity = ['stability_type', 'temperature', 'time', 'humidity', 'efficiency_cont', 'efficiency_tret']

    # Initialize comparison dictionaries
    text_dict = {var: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for var in outside_variables}
    stability_dict = {var: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for var in stability_entity}

    for row in df.itertuples():       
        label_value = row.annotation
        extracted_value = row.extracted

        # print(label_value)
        # print(extracted_value)

        for id, label in label_value.items():
            if ('test' in id) and (isinstance(label_value[id], dict)):
                ##Plan for stability test evaluation
                '''
                For each stability condition in annotation, 
                    Pair them with stability condition in extracted
                        With stability of annotation and extraction, use function tests_comparison that returns how similar 2 stabilities are
                    
                Once all the pair is calculated, find the stability name of extraction that was closest to annotation stability. 

                Using this dictionary, we will increment FN, FP, TN, TP for each element of the entity.
                '''
                matched = 0
                stability_match = {}
                for extract_id, extract_label in extracted_value.items():
                    if ('test' in extract_id) and (isinstance(extracted_value[extract_id], dict)):
                        matched += 1
                        match_list = tests_comparison(id, label, extract_id, extract_label)
                        match_list = [0 if item is None else item for item in match_list]
                        # print(extracted_value[extract_id])
                        # print(match_list)
                        stability_match[extract_id] = match_list
        
                if matched == 0:
                    #No stability were extracted, we will add stability_unmatched
                        ##We need to account for if there was NO stability extracted. 
                    for key in stability_dict:
                        if 'FN' in stability_dict[key]:
                            stability_dict[key]['FN'] += 1
                else:
                    stability_match_mean = {stability: np.mean(lis) for stability, lis in stability_match.items()}
                    max_key = max(stability_match_mean, key=stability_match_mean.get)  
                    # print(extracted_value[max_key])
                    ##Now, I need to compare each entity in that found max_key and fill in that FN, dictionary.
                    for entity in label_value[id].keys():
                        if entity == 'efficiency_control':
                            continue
                        if entity == 'perovskite_molecule':
                            continue
                        entity_result = entity_comparison(entity, label, extracted_value[max_key])
                        stability_dict[entity][entity_result] += 1  
            else:  
                if id in outside_text:
                    if (id in label_value) and (id in extracted_value):
                        result = text_comparison(id, label_value, extracted_value)
                        text_dict[id][result] += 1
                    else:
                        print(f"This id {id} was not in extracted dictionary")
                        if label_value[id] == None:
                            text_dict[id]["TP"] += 1
                        else:
                            text_dict[id]["FN"] += 1
                        
                        print('annotation ',label_value)
                        print('extraction ',extracted_value)
                else:
                    if (id in label_value) and (id in extracted_value):
                        result = numeric_comoparison(id, label_value, extracted_value)
                        text_dict[id][result] += 1
                    else:
                        print(f"This id {id} was not in extracted dictionary")
                        if label_value[id] == None:
                            text_dict[id]["TP"] += 1
                        else:
                            text_dict[id]["FN"] += 1
                        
                        print('annotation ',label_value)
                        print('extraction ',extracted_value)



    # Merge all results
    combined_dict = {**text_dict, **stability_dict}
    # print("Performance for each variable in dictionary:", combined_dict)

    # Compute precision, recall, and F1-score
    variable_list, precision_list, recall_list, f1_list, f1_dict = [], [], [], [], {}
    for variable, performance in combined_dict.items():
        TP, FP, FN = performance["TP"], performance["FP"], performance["FN"]
        
        precision = safe_division(TP, TP + FP)
        recall = safe_division(TP, TP + FN)
        f1 = safe_division(2 * precision * recall, precision + recall)

        variable_list.append(variable)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        f1_dict[variable] = f1

    return combined_dict, variable_list, precision_list, recall_list, f1_list, f1_dict

dict_result_base, variables_base, precisions_base, recalls_base, f1s_base, f1_dict_base = compare_json(evaluate_df_base)
dict_result_deepseek, variables_deepseek, precisions_deepseek, recalls_deepseek, f1s_deepseek, f1_dict_deepseek = compare_json(evaluate_df_deepseek)
dict_result_llama, variables_llama, precisions_llama, recalls_llama, f1s_llama, f1_dict_llama = compare_json(evaluate_df_llama)

## Performed MACRO F1 Score
def weight_dict(keys, weights):

    weights_dictionary = {}
    for i in range(len(weights)):
        weights_dictionary[keys[i]] = weights[i]

    return weights_dictionary

def macro_f1(f1_dict, weight = None):
    if weight == None:
        # print(weight)
        #If no weight given, do unweighted average of f1 score
        return sum(f1_dict) / len(f1_dict)
    else:
        total_f1 = 0
        sum_weight = 0
        for key in f1_dict.keys():
            total_f1 += (f1_dict[key] * weight[key])
            sum_weight += (weight[key])
        # for i in range(len(f1_list)):
        #     total_f1 += (f1_list[i] * weight[i])

        return total_f1 / sum_weight
    
# Define column names
columns = ['Macro F1 score weight distribution', 'DeepSeek R1 8B', 'DeepSeek R1 8B Finetuned', 'Llama-3.2 3B Instruct']
df_f1scores = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4, 5], columns=columns)
macro_base_0 = macro_f1(f1s_base)
macro_train_0 = macro_f1(f1s_deepseek)
macro_llama_0 = macro_f1(f1s_llama)
unweighted = ['Macro F1 score with equal weight', macro_base_0, macro_train_0, macro_llama_0]
df_f1scores.loc[0] = unweighted

keys = ['perovskite_composition',
 'electron_transport_layer',
 'hole_transport_layer',
 'structure_pin_nip',
 'stability_type',
 'temperature',
 'time',
 'humidity',
 'passivating_molecule',
 'efficiency_cont',
 'efficiency_tret',
 'control_pce',
 'treated_pce',
 'control_voc',
 'treated_voc']
weights_1_list = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1]
weights_1 = weight_dict(keys, weights_1_list)
macro_base_1 = macro_f1(f1_dict_base, weight = weights_1)
macro_train_1 = macro_f1(f1_dict_deepseek, weights_1)
macro_llama_1 = macro_f1(f1_dict_llama, weight = weights_1)
first_f1 = ['Heavier weight on stability', macro_base_1, macro_train_1, macro_llama_1]
df_f1scores.loc[1] = first_f1

weights_2_list = [2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
weights_2 = weight_dict(keys, weights_2_list)
macro_base_2 = macro_f1(f1_dict_base, weight = weights_2)
macro_train_2 = macro_f1(f1_dict_deepseek, weight = weights_2)
macro_llama_2 = macro_f1(f1_dict_llama, weight = weights_2)
first_f2 = ['Heavier weight on perovskite structure', macro_base_2, macro_train_2, macro_llama_2]
df_f1scores.loc[2] = first_f2

weights_3_list = [1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2]
weights_3 = weight_dict(keys, weights_3_list)
macro_base_3 = macro_f1(f1_dict_base, weight = weights_3)
macro_train_3 = macro_f1(f1_dict_deepseek, weight = weights_3)
macro_llama_3 = macro_f1(f1_dict_llama, weight = weights_3)
first_f3 = ['Heavier weight on numeric data', macro_base_3, macro_base_3, macro_llama_3]
df_f1scores.loc[3] = first_f3

weights_4_list = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
weights_4 = weight_dict(keys, weights_4_list)
macro_train_4 = macro_f1(f1_dict_deepseek, weight = weights_4)
macro_base_4 = macro_f1(f1_dict_base, weight = weights_4)
macro_llama_4 = macro_f1(f1_dict_llama, weight = weights_4)

weights_5_list = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
weights_5 = weight_dict(keys, weights_5_list)
macro_base_5 = macro_f1(f1_dict_base, weight = weights_5)
macro_train_5 = macro_f1(f1_dict_deepseek, weight = weights_5)
macro_llama_5 = macro_f1(f1_dict_llama, weight = weights_5)
first_f5 = ['Veriable to perform prediction 1: Normalized difference in PCE', macro_base_5, macro_train_5, macro_llama_5]
df_f1scores.loc[4] = first_f5

weights_6_list = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
weights_6 = weight_dict(keys, weights_6_list)
macro_train_6 = macro_f1(f1_dict_deepseek, weight = weights_6)
macro_base_6 = macro_f1(f1_dict_base, weight = weights_6)
macro_llama_6 = macro_f1(f1_dict_llama, weight = weights_6)
first_f6 = ['Veriable to perform prediction 2: Prediction of long term PCE retained', macro_base_6, macro_train_6, macro_llama_6]
df_f1scores.loc[5] = first_f6

df_f1scores.to_csv('evaluation_schema_2.csv', index=False)


