## This file compares teamtat annotation with extraction performed on Original Schema

from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import numpy as np
import json
import os
import xml.etree.ElementTree as ET 
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


## Numerical list of relevent and irrelevent numbers in 0-149 papers
irrelevent_papers = [26,28,29,30,32,33,34,35,38,43,44,45,52,54,55,56,57,58,63,66,68,69,70,78,80,83,84,86,87,88,89,90,91,92,93,94,98,100,101,102,103,104,105,106,108,109,110,111,112,115,116,117,119,121,125,128,129,130,134,136,138,139, 140]
relevent_bad = [1, 18, 20, 25, 27, 41, 51, 61, 71, 76, 135, 141, 145]

relevent_good = [i for i in range(0, 150) if i not in irrelevent_papers and i not in relevent_bad]

#### File preparation to ensure that both annotation and extraction are in correct JSON

def str_toJson(string):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    if string == None:
        return None
    json_string = string.replace("None", "null")

    try:
        # Try to load the JSON string
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        # Catch JSONDecodeError if the string is not valid JSON
        print(f"Error decoding JSON: {e}")
        print(string)
        print(json_string)
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return None
    
def include_passivating(dictionary):
    ##In extraction json, realized that some extraction has passivating molecule that is NOT included in its stability testing. 
    ## Since passivating molecule (if exist) needs to be in stability testing (nexted dictionary), we will transfer the information and spit out a cleaned dictionary. 
    # print(dictionary)
    if dictionary == None:
        return None
    if "passivating_molecule" in dictionary.keys():
        passivating = dictionary['passivating_molecule']
        del dictionary['passivating_molecule']
        
        for entity in dictionary.keys():
            if entity.startswith('test'):
                # print(i['entity'])
                if type(dictionary[entity]) == dict:
                    if 'passivating_molecule' in dictionary[entity].keys():
                        continue
                    else:
                        # print("Have to include passivating molecule in tests")
                        dictionary[entity]['passivating_molecule'] = passivating
        
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
                        # else:
                            
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
                        # else:
                            
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
    ##Change the efficiency value to be in percent
    if dictionary == None:
        return None
    entity_decimal = ['efficiency_cont','efficiency_tret']
    for key in dictionary.keys():
        if (key.startswith('test')) & (type(dictionary[key]) == dict):
            for entity in dictionary[key].keys():
                if (entity in entity_decimal) and (dictionary[key][entity] != None):
                    if dictionary[key][entity] == dictionary[key][entity] > 1:
                        dictionary[key][entity] = dictionary[key][entity] / 100
    return dictionary

def convert_efficiency_key(dictionary):
    ## Some extraction dictionary had inconsistent keys that needs to be converted. 
    if dictionary == None:
        return None
    for key, item in dictionary.items():
        if 'test' in key:
            if item == None:
                continue
            if ("efficiency_cont" in dictionary[key].keys()) | ("efficiency_tret" in dictionary[key].keys()):
                continue
            else:
                if 'retained_proportion_cont' in dictionary[key]:
                    dictionary[key]['efficiency_cont'] = dictionary[key].pop('retained_proportion_cont')
                if 'retained_proportion_tret' in dictionary[key]:
                    dictionary[key]['efficiency_tret'] = dictionary[key].pop('retained_proportion_tret')
                if 'retained_percentage_tret' in dictionary[key]:
                    dictionary[key]['efficiency_tret'] = dictionary[key].pop('retained_percentage_tret')
                if 'retained_percentage_cont' in dictionary[key]:
                    dictionary[key]['efficiency_cont'] = dictionary[key].pop('retained_percentage_cont')
                if 'control_efficiency' in dictionary[key]:
                    dictionary[key]['efficiency_cont'] = dictionary[key].pop('control_efficiency')
                if 'treatment_efficiency' in dictionary[key]:
                    dictionary[key]['efficiency_tret'] = dictionary[key].pop('treatment_efficiency')      
    return dictionary

def convert_structure_key(dictionary):
    ## Some extraction dictionary had inconsistent NIP PIN key that needs to be converted. 
    if dictionary == None:
        return None
    found = 0
    founded = 0
    foundd = 0
    for key, item in dictionary.items():
        if key == "pin_nip_structure":
            found = 1
        if key == "structure_type":
            founded = 1
        if key == "structure":
            foundd = 1
    if found == 1:
        dictionary['structure_pin_nip'] = dictionary['pin_nip_structure']
        dictionary.pop('pin_nip_structure')
    if founded == 1:
        dictionary['structure_pin_nip'] = dictionary['structure_type']
        dictionary.pop('structure_type')
    if foundd == 1:
        dictionary['structure_pin_nip'] = dictionary['structure']
        dictionary.pop('structure')


    return dictionary

def escape_internal_quotes(json_string):
    # This regex finds values within quotes and fixes internal unescaped quotes
    return re.sub(r'":\s*"([^"]*?)"', lambda m: '": "' + m.group(1).replace('"', '\\"') + '"', json_string)

def force_fix(json_string):
    '''
    The quotateion mark are forbidding string output to be converted to JSON, 
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

def str_toJson_8bit(string):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    if string == None:
        return None
    
    json_string = string.replace("None", "null")
    json_string = json_string.replace(r'\"', '"')
    # json_string = json_string.replace("'", '"')
    json_string = escape_internal_quotes(json_string)
    json_string = force_fix(json_string)

    json_string = json_string.replace("',\n", '",')
    json_string = json_string.replace("\n", '')
    json_string = json_string.replace("'  }", '"}')
    


    try:
        # Try to load the JSON string
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        # Catch JSONDecodeError if the string is not valid JSON
        print("Error decoding 8bit deepseek finetuned")
        print(f"Error decoding JSON: {e}")
        print(json_string)
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return None
    
def str_toJson_llama(strings):
    ##The json output from annotation dataframe was not in correct json format
    # We will change the None to null
    if strings == None:
        return None
    json_string = strings.replace("None", "null")
    json_string = json_string.replace(r'\"', '"')
    json_string = escape_internal_quotes(json_string)
    json_string = json_string.replace("'", '"')
    json_string = force_fix(json_string)
    
    json_string = json_string.replace("True", "true")
    try:
        # Try to load the JSON string
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        # Catch JSONDecodeError if the string is not valid JSON
        print("Error decoding llama")
        print(f"Error decoding JSON: {e}")
        print(json_string)
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return None


### Loading in TEAMTAT ANNOTATION as dataframe
#Teamtat Annotation
annotation_df = pd.read_csv("data/150_papers_json_update.csv")[["id", "first_num", "output"]]
annotation_df = annotation_df.sort_values(by = ['first_num'])
##Change the output column to be converted to json
annotation_df['output'] = annotation_df['output'].apply(str_toJson)
annotation_df['output'] = annotation_df['output'].apply(convert_numeric)
## Get the annotation_df with only relevent good papers. 
annotation_df = annotation_df[annotation_df['first_num'].isin(relevent_good)]



### Loading in Extraction perfomed by the basemodel
with open("data/output1.json", 'r') as f:
    extraction = json.load(f)

extraction_base = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_base['paper_num'] = pd.to_numeric(extraction_base['paper_num'])
extraction_base = extraction_base.sort_values('paper_num')
extraction_base['output'] = extraction_base['output'].apply(include_passivating)
extraction_base['output'] = extraction_base['output'].apply(convert_numeric)
extraction_base['output'] = extraction_base['output'].apply(convert_efficiency)


## Loading in Extraction perfomed by the finetuned deepseek 4 bit
with open("annotation/Original_schema/data/deepseek_finetuned_4bit.json", 'r') as f:
    extraction = json.load(f)

extraction_train = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_train['paper_num'] = pd.to_numeric(extraction_train['paper_num'])
extraction_train = extraction_train.sort_values('paper_num')
extraction_train['output'] = extraction_train['output'].apply(include_passivating)
extraction_train['output'] = extraction_train['output'].apply(convert_efficiency)
extraction_train['output'] = extraction_train['output'].apply(convert_efficiency_key)
extraction_train['output'] = extraction_train['output'].apply(convert_structure_key)
extraction_train['output'] = extraction_train['output'].apply(convert_numeric)

## extraction performed by finetuned deepseek 8 bit
with open("annotation/Original_schema/data/deepseek_8bit_finetuned.json", 'r') as f:
    extraction = json.load(f)

extraction_train_8 = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
extraction_train_8['paper_num'] = pd.to_numeric(extraction_train_8['paper_num'])
extraction_train_8 = extraction_train_8.sort_values('paper_num')
extraction_train_8['output'] = extraction_train_8['output'].apply(str_toJson_8bit)
extraction_train_8['output'] = extraction_train_8['output'].apply(include_passivating)
extraction_train_8['output'] = extraction_train_8['output'].apply(convert_efficiency_key)
extraction_train_8['output'] = extraction_train_8['output'].apply(convert_structure_key)
extraction_train_8['output'] = extraction_train_8['output'].apply(convert_numeric)

## extraction performed by Llama
with open("annotation/Original_schema/data/llama_3b_8bit_fully_nested.json", 'r') as f:
    extraction = json.load(f)

llama = pd.DataFrame(list(extraction.items()), columns=['paper_num', 'output'])
llama['paper_num'] = pd.to_numeric(llama['paper_num'])
llama = llama.sort_values('paper_num')
llama['output'] = llama['output'].apply(str_toJson_llama)
llama['output'] = llama['output'].apply(include_passivating)
llama['output'] = llama['output'].apply(convert_efficiency_key)
llama['output'] = llama['output'].apply(convert_structure_key)
llama['output'] = llama['output'].apply(convert_numeric)


## Merging annotation with extraction to create single df

#Merging the base extraction
evaluate_df_base = annotation_df.merge(extraction_base, left_on='first_num', right_on='paper_num', how = 'left')[["first_num", "output_x",'output_y']]
evaluate_df_base.columns = ['paper_num', 'annotation', 'extracted']
evaluate_df_base["extracted"] = evaluate_df_base["extracted"].apply(lambda x: None if pd.isna(x) else x)
evaluate_df_base_absent = evaluate_df_base[evaluate_df_base['extracted'].isnull()]
    ##Dropping row with None In extracted column
evaluate_df_base = evaluate_df_base[evaluate_df_base['extracted'].notnull()]

#Merging the finetuned deepseek 4 bit
evaluate_df_train = annotation_df.merge(extraction_train, left_on='first_num', right_on='paper_num', how = 'left')[["first_num", "output_x",'output_y']]
evaluate_df_train.columns = ['paper_num', 'annotation', 'extracted']
evaluate_df_train["extracted"] = evaluate_df_train["extracted"].apply(lambda x: None if pd.isna(x) else x)
evaluate_df_train_absent = evaluate_df_train[evaluate_df_train['extracted'].isnull()]
    ##This code below is for dropping row with None In extracted
evaluate_df_train = evaluate_df_train[evaluate_df_train['extracted'].notnull()]

#Merging the finetuned deepseek 8 bit
evaluate_df_train8 = annotation_df.merge(extraction_train_8, left_on='first_num', right_on='paper_num')[["paper_num", "output_x",'output_y']]
evaluate_df_train8.columns = ['paper_num', 'annotation', 'extracted']
evaluate_df_train8_absent = evaluate_df_train8[evaluate_df_train8['extracted'].isnull()]
    ##This code below is for dropping row with None In extracted
evaluate_df_train8 = evaluate_df_train8[evaluate_df_train8['extracted'].notnull()]

#Merging the Llama 
evaluate_df_llama = annotation_df.merge(llama, left_on='first_num', right_on='paper_num')[["paper_num", "output_x",'output_y']]
evaluate_df_llama.columns = ['paper_num', 'annotation', 'extracted']
evaluate_df_llama_absent = evaluate_df_llama[evaluate_df_llama['extracted'].isnull()]
    ##This code below is for dropping row with None In extracted
evaluate_df_llama = evaluate_df_llama[evaluate_df_llama['extracted'].notnull()]



##### EVALUATION IMPLEMENTATION of comparing annotation JSON with extraction JSON
def tests_comparison(stability_annotated, label_dict, stability_extracted, extract_dict):
    '''
    Given pair of stability test results from annotation and extraction, this function returns a metric of how similart these two test results were
    This is used to find the best pair of stability test between the labeled stability to extraction stability
    '''
    # print(stability_annotated, label_dict, stability_extracted, extract_dict)
    stability_entity_annotated = ['stability_type', 'passivating_molecule', 'temperature', 'time', 'humidity', 'efficiency_cont', 'efficiency_tret', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    stability_entity_extracted = ['test_name', 'passivating_molecule', 'temperature', 'time', 'humidity', 'control_efficiency', 'treatment_efficiency', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']   
    # print(f"stability_annotated{stability_annotated}")
    # print(f"label_dict{label_dict}")
    # print(f"stability_extracted{stability_extracted}")
    # print(f"extract_dict{extract_dict}")

    compared_metric = []
    numeric_data_annotated = []
    numeric_data_extracted = []
    for entity_i in range(len(stability_entity_annotated)):
        if entity_i <= 1:
            if stability_entity_extracted[entity_i] not in extract_dict.keys():
                extract_dict[stability_entity_extracted[entity_i]] = None

            if (label_dict[stability_entity_annotated[entity_i]] == None) | (extract_dict[stability_entity_extracted[entity_i]] == None):
                compared_metric.append(None)
            else:
                ##Text entity, perform Sequence Matcher 
                compared = SequenceMatcher(None, label_dict[stability_entity_annotated[entity_i]], extract_dict[stability_entity_extracted[entity_i]]).ratio()
                # print(compared)
                if entity_i == 0:
                    if compared > 0.9:
                        compared_metric.append(1)
                    else:
                        compared_metric.append(0)
                else:
                    compared_metric.append(compared)
        else:
            if stability_entity_extracted[entity_i] not in extract_dict.keys():
                extract_dict[stability_entity_extracted[entity_i]] = 0
            elif extract_dict[stability_entity_extracted[entity_i]] == None:
                extract_dict[stability_entity_extracted[entity_i]] = 0

            if stability_entity_annotated[entity_i] not in label_dict.keys():
                label_dict[stability_entity_annotated[entity_i]] = 0
            elif label_dict[stability_entity_annotated[entity_i]] == None:
                label_dict[stability_entity_annotated[entity_i]] = 0

                
            numeric_data_annotated.append(label_dict[stability_entity_annotated[entity_i]])
            numeric_data_extracted.append(extract_dict[stability_entity_extracted[entity_i]])

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
    This function actually comparied each entity in stability testing once the best pair was determined
    Return TP, TN, FP, FN

    The similarity threshold motivations
    The tolarance of 2.7% was what was reasonable looking at the absolute difference
    treated_voc 1.18, 1.149, absolute difference 0.026271186440677895

    The text similarity were set to 75% due to the structure example
    FP, NIP, n-i-p, 0.75
    This should be positive
    
    '''
    text_entity = ['stability_type', 'passivating_molecule']
    numerical_entity = ['time', 'efficiency_cont', 'efficiency_tret', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    numerical_exception = ['temperature', 'humidity']

    if entity in text_entity:
        key_to_check = "test_name" if entity == "stability_type" else entity

        # If the key is missing in the extracted annotation, return False Negative
        if (label[entity]!=None) & (extracted_dict[key_to_check]==None):
            # print(f"FN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
            return "FN"
        elif (label[entity]==None) & (extracted_dict[key_to_check]!=None):
            # print(f"TN, {label_annotation[id]}, {extraction_annotation[key_to_check]}")
            return "TN"

        label_data = label.get(entity, "")
        extract_data = extracted_dict.get(key_to_check, "")

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

        # print(f"annotated{label[entity]}")
        # print(f"extracted{extracted_dict[entity]}")
        # print(entity)
        # print(extracted_dict)
        if entity not in extracted_dict.keys():
            extracted_dict[entity] = 0
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

        # print(entity)
        # print(type(extracted_dict[entity]))
        # print(label[entity])
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
            
            # print(entity)
            # print(type(extracted_dict[entity]))
            # print(extracted_dict)
            # print(label)
            if type(extracted_dict[entity]) == str:
                ## This is a case where the extraction was string in range of number and annotation were a number. 
                extracted_first_num = float(extracted_dict[entity][:2])
                extracted_second_num = float(extracted_dict[entity][3:])
                if extracted_first_num < label[entity] < extracted_second_num:
                    return "TP"
                else:
                    return "FP"

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
                    ### This is where the data is value+-ME
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
                    ### 30-50 
                    # print(label[entity], type(label[entity]))
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

def text_comparison(id, label_annotation, extraction_annotation, text_similarity_threshold=0.75):
    """Compares text values using string similarity matching.
    - THE 4 basic variable that is to compare is PEROVSKITE COMPOSITION, ETL, HTL, STRUCTURE
    """

    # Handle special case for structure_pin_nip
    # key_to_check = "pin_nip_structure" if id == "structure_pin_nip" else id
    # print(id)
    # print(extraction_annotation)
    if extraction_annotation == None:
        return "FN"
    # print(id)
    # print(extraction_annotation)
    # If the key is missing in the extracted annotation, return False Negative
    if (id not in extraction_annotation.keys()):
        return "FN"

    if (extraction_annotation[id]==None):
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
    
def compare_json(df):
    """
    Compare labeled and extracted JSON data for correctness.

    TP: Correct value extracted by LLM.
    FN: LLM didn't extract this variable.
    FP: LLM extracted a value, but it was incorrect.
    TN: LLM halucinated and returned value that was not extracted
    """
    
    text_variables = ['perovskite_composition', 'electron_transport_layer', 'hole_transport_layer', 'structure_pin_nip']

    
    stability_entity_annotated = ['stability_type', 'temperature', 'time', 'humidity', 'passivating_molecule', 'efficiency_cont', 'efficiency_tret', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    stability_entity_extracted = ['test_name', 'temperature', 'time', 'humidity', 'passivating_molecule','control_efficiency', 'treatment_efficiency', 'control_pce', 'treated_pce', 'control_voc', 'treated_voc']
    
    # Initialize comparison dictionaries
    text_dict = {var: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for var in text_variables}
    stability_dict = {var: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for var in stability_entity_annotated}

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
                
                if extracted_value != None:
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
                    ##Max key is a test_... KEY in Extraction that BEST matched the testID in annotation. 
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
                result = text_comparison(id, label_value, extracted_value)
                
                text_dict[id][result] += 1


    # Merge all results
    combined_dict = {**text_dict, **stability_dict}
    # print("Performance for each variable in dictionary:", combined_dict)

    # Compute precision, recall, and F1-score
    variable_list, precision_list, recall_list, f1_list = [], [], [], []
    for variable, performance in combined_dict.items():
        TP, FP, FN = performance["TP"], performance["FP"], performance["FN"]
        
        precision = safe_division(TP, TP + FP)
        recall = safe_division(TP, TP + FN)
        f1 = safe_division(2 * precision * recall, precision + recall)

        variable_list.append(variable)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return combined_dict, variable_list, precision_list, recall_list, f1_list

## Actually comparing the dictionaries
dict_result_base, variables_base, precisions_base, recalls_base, f1s_base = compare_json(evaluate_df_base)
dict_result_train, variables_train, precisions_train, recalls_train, f1s_train = compare_json(evaluate_df_train)
dict_result_train_8, variables_train_8, precisions_train_8, recalls_train_8, f1s_train_8 = compare_json(evaluate_df_train8)
dict_result_llama, variables_llama, precisions_llama, recalls_llama, f1s_llama = compare_json(evaluate_df_llama)



## Calculating Macro f1 score (weighted f1 score)
"""
F1 score are calculated on 6 different criteria
- Unweight
- Heavier weight on stability
- Heavier weight on perovskite structure
- Heavier weight on numeric data
- Weight to perform prediction 0: Predicting treated PCE
- Weight to perform prediction 1: Predicting the normalized PCE difference between treated and control PCE
- Weight to perform prediction 2: Predicting the percent PCE retained after stability testing
"""
def macro_f1(f1_list, weight = None):
    if weight == None:
        #If no weight given, do unweighted average of f1 score
        return sum(f1_list) / len(f1_list)
    total_f1 = 0
    for i in range(len(f1_list)):
        total_f1 += (f1_list[i] * weight[i])
    return total_f1 / sum(weight)

# Define column names
columns = ['Macro F1 score weight distribution', 'DeepSeek R1 8B', 'DeepSeek R1 4B Finetuned', 'DeepSeek R1 8B Finetuned', 'Llama-3.2 3B Instruct']
# Df to store all the results
df_f1scores = pd.DataFrame(np.nan, index=[0, 1, 2, 3, 4, 5, 6], columns=columns)

## Unweighted
macro_base_0 = macro_f1(f1s_base)
macro_train_0 = macro_f1(f1s_train)
macro_train8_0 = macro_f1(f1s_train_8)
macro_llama_0 = macro_f1(f1s_llama)
unweighted = ['Macro F1 score with equal weight', macro_base_0, macro_train_0, macro_train8_0, macro_llama_0]
df_f1scores.loc[0] = unweighted

##Weight 1
weights_1 = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1]
macro_train_1 = macro_f1(f1s_train, weight = weights_1)
macro_base_1 = macro_f1(f1s_base, weight = weights_1)
macro_train8_1 = macro_f1(f1s_train_8, weight = weights_1)
macro_llama_1 = macro_f1(f1s_llama, weight = weights_1)
first_f1 = ['Heavier weight on stability', macro_base_1, macro_train_1, macro_train8_1, macro_llama_1]
df_f1scores.loc[1] = first_f1

##Weight 2
weights_2 = [2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
macro_train_2 = macro_f1(f1s_train, weight = weights_2)
macro_base_2 = macro_f1(f1s_base, weight = weights_2)
macro_train8_2 = macro_f1(f1s_train_8, weight = weights_2)
macro_llama_2 = macro_f1(f1s_llama, weight = weights_2)
first_f2 = ['Heavier weight on perovskite structure', macro_base_2, macro_train_2, macro_train8_2, macro_llama_2]
df_f1scores.loc[2] = first_f2

##Weight 3
weights_3 = [1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2]
macro_train_3 = macro_f1(f1s_train, weight = weights_3)
macro_base_3 = macro_f1(f1s_base, weight = weights_3)
macro_train8_3 = macro_f1(f1s_train_8, weight = weights_3)
macro_llama_3 = macro_f1(f1s_llama, weight = weights_3)
first_f3 = ['Heavier weight on numeric data', macro_base_3, macro_train_3, macro_train8_3, macro_llama_3]
df_f1scores.loc[3] = first_f3

##Weight 4
weights_4 = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
macro_train_4 = macro_f1(f1s_train, weight = weights_4)
macro_base_4 = macro_f1(f1s_base, weight = weights_4)
macro_train8_4 = macro_f1(f1s_train_8, weight = weights_4)
macro_llama_4 = macro_f1(f1s_llama, weight = weights_4)
first_f4 = ['Weight to perform prediction 1', macro_base_4, macro_train_4, macro_train8_4, macro_llama_4]
df_f1scores.loc[4] = first_f4

##Weight 5
weights_5 = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
macro_train_5 = macro_f1(f1s_train, weight = weights_5)
macro_base_5 = macro_f1(f1s_base, weight = weights_5)
macro_train8_5 = macro_f1(f1s_train_8, weight = weights_5)
macro_llama_5 = macro_f1(f1s_llama, weight = weights_5)
first_f5 = ['Weight to perform prediction 2', macro_base_5, macro_train_5, macro_train8_5, macro_llama_5]
df_f1scores.loc[5] = first_f5

##Weight 6
weights_6 = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
macro_train_6 = macro_f1(f1s_train, weight = weights_6)
macro_base_6 = macro_f1(f1s_base, weight = weights_6)
macro_train8_6 = macro_f1(f1s_train_8, weight = weights_6)
macro_llama_6 = macro_f1(f1s_llama, weight = weights_6)
first_f6 = ['Weight to perform prediction 3', macro_base_6, macro_train_6, macro_train8_6, macro_llama_6]
df_f1scores.loc[6] = first_f6

df_f1scores.to_csv("../data/extraction_eval/f1_scores_originalschema.csv", index = False)