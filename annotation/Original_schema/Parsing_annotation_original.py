## This file parces the Annotation BIOC and store the annotation content as a CSV to feed into evaluation_original.py
## The exported csv file is named 150_papers_json_update.csv

import json
import os
import xml.etree.ElementTree as ET
import pandas as pd

EMPTY_PAPER_DATA = {
    "perovskite_composition": None,
    "electron_transport_layer": None,
    "hole_transport_layer": None,
    "structure_pin_nip": None,
}
EMPTY_STABILITY_TEST = {
    "stability_type": None,
    "passivating_molecule": None,
    "humidity": None,
    "temperature": None,
    "time": None,
    "control_pce": None,
    "treated_pce": None,
    "control_voc": None,
    "treated_voc": None,
    "efficiency_cont": None,
    "efficiency_tret": None
}

def get_json_for_passage(passage, relations, previous_json):
    concept_ids = set()
    for annotation in passage.findall(".//annotation"):
        node_id = annotation.get("id")
        var_name = annotation.find("infon[@key='type']").text
        concept_id = annotation.find("infon[@key='identifier']").text
        value = annotation.find("text").text
        value = concept_id if concept_id is not None else value
        # if var_name == "perovskite_molecule": #due to an error in some of the annotations
        #     var_name = "passivating_molecule"
        if var_name in ["additive_molecule", "treatment_element", "control_element", "metal_contact"]: #irrelevant
            continue

        if var_name in ["perovskite_composition", "structure_pin_nip", "electron_transport_layer", "hole_transport_layer" ]:
            #in top level: composition, ETL, HTL, PIN-NIP,
            previous_json[var_name] = value
        elif node_id in relations:
            test_names = relations[node_id]
            for test_name in test_names:
                if test_name not in previous_json:
                    previous_json[test_name] = EMPTY_STABILITY_TEST.copy()
                previous_json[test_name][var_name] = value
        elif len(relations.keys()) == 0:
            if "test_1" not in previous_json:
                previous_json["test_1"] = EMPTY_STABILITY_TEST.copy()
            previous_json["test_1"][var_name] = value
            #in stability tests:
            #test type, passivator, PCE (control + treat), VOC (control + treat)
            #efficiency (treat, control), temp, time, humidity
        else:
            #assumes that all other possible data goes into the first stability test
            if "test_1" not in previous_json:
                previous_json["test_1"] = EMPTY_STABILITY_TEST.copy()
            previous_json["test_1"][var_name] = value

    return previous_json

def extract_papernum(root):
    first_text = root.find(".//text")
    full_text = first_text.text
    
    ##We want to extract article number from this format
    #Method: split by spaces and extract the last element in the list
    text_list = full_text.split()
    paper_num = text_list[-1]
    return paper_num

def parse_bioc_into_chunks(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    passages = root.findall('.//passage')
    data = []

    relations = {}
    test_names = set()
    for relation in root.findall(".//relation"):
        test_name = relation.find("infon[@key='type']").text
        if 'performance' in test_name: #irrelevant tests
            continue
        if test_name not in test_names:
            test_names.add(test_name)
        else:
            test_name = test_name + "_2"
        node_ids = [node.get("refid") for node in relation.findall("node")]
        for node_id in node_ids:
            if node_id not in relations:
                relations[node_id] = [test_name]
            else:
                relations[node_id].append(test_name)

    paper_num = extract_papernum(root)
    curr_json = EMPTY_PAPER_DATA.copy()
    for relation in root.findall('.//relation'):
        test_name = relation.find
    for i, passage in enumerate(passages):
        passage_text = passage.find('.//text').text
        row = { "id": f"{paper_num}_{i}", "text": passage_text, "memory": json.dumps(curr_json) }
        curr_json = get_json_for_passage(passage, relations, curr_json)
        row['output'] = json.dumps(curr_json)
        data.append(row)
    return data

def get_numbers(string):
    if ":" in string:
        return string.split(":")[1]
    else:
        return string
    
### LOADING THE ANNOTATION BIOC
bioc_dir = "../biocs"
data = []
for filename in os.listdir(bioc_dir):
    if filename.endswith(".xml"):
        file_path = os.path.join(bioc_dir, filename)
        curr_paper_chunks = parse_bioc_into_chunks(file_path)
        data.extend(curr_paper_chunks)

df = pd.DataFrame(data)
df[['first_num', 'second_num']] = df['id'].str.split('_', expand=True)

# print(df['first_num'])
# Step 2: Convert 'first_num' to numeric for proper sorting
df['first_num'] = df['first_num'].apply(get_numbers)
df['first_num'] = df['first_num'].astype(int)

# Step 3: Group by 'first_num' and get the last row of each group
result = df.groupby('first_num', as_index=False).last()
result.to_csv('data/150_papers_json_update.csv', index=False)
