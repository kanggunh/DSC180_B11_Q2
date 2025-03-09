## This file is used to create annotations_flattened.json for Schema 2 and Schema 3 evaluation


import json
import os
import xml.etree.ElementTree as ET
import pandas as pd
import re

EMPTY_PAPER_DATA = {
    "perovskite_composition": None,
    "electron_transport_layer": None,
    "hole_transport_layer": None,
    "structure_pin_nip": None,
    "passivating_molecule": None,
    "control_pce": None,
    "treated_pce": None,
    "control_voc": None,
    "treated_voc": None,
}
EMPTY_STABILITY_TEST = {
    "stability_type": None,
    "humidity": None,
    "temperature": None,
    "time": None,
    "efficiency_cont": None,
    "efficiency_tret": None
}

def extract_papernum(root):
    first_text = root.find(".//text")
    full_text = first_text.text
    
    ##We want to extract article number from this format
    #Method: split by spaces and extract the last element in the list
    text_list = full_text.split()
    paper_num = text_list[-1]
    numbers = ''.join(filter(str.isdigit, paper_num))
    return numbers 

def get_passivators(root):
    passivators = {}
    for annotation in root.findall(".//annotation"):
        node_id = annotation.get("id")
        var_name = annotation.find("infon[@key='type']").text
        concept_id = annotation.find("infon[@key='identifier']").text
        value = annotation.find("text").text
        value = concept_id if concept_id else value
        if var_name != "passivating_molecule" or value == None:
            continue
        
        if value in passivators:
            passivators[value].append(node_id)
        else:
            passivators[value] = [node_id]
    return passivators

def get_relations(root):
    relations = {}
    for relation in root.findall(".//relation"):
        test_name = relation.find("infon[@key='type']").text
        relation_id = relation.get("id")
        if 'performance' in test_name:
            continue
        node_ids = [node.get("refid") for node in relation.findall("node")]
        for node_id in node_ids:
            if node_id not in relations:
                relations[node_id] = [relation_id]
            else:
                relations[node_id].append(relation_id)
    return relations

def get_json_for_paper(root, passivators, relations):
    output = {}
    if len(passivators) == 0:
        return None
    for passivator in passivators:
        relevant_tests = []
        curr_object = EMPTY_PAPER_DATA.copy()
        passivator_nodes = passivators[passivator]
        for node_id in passivator_nodes:
            if node_id in relations:
                relevant_tests = relations[node_id]
        for annotation in root.findall(".//annotation"):
            node_id = annotation.get("id")
            var_name = annotation.find("infon[@key='type']").text
            concept_id = annotation.find("infon[@key='identifier']").text
            value = annotation.find("text").text
            value = concept_id if concept_id is not None else value
            if var_name in ["additive_molecule", "treatment_element", "control_element", "metal_contact", "perovskite_molecule"]: #irrelevant
                continue

            if var_name in ["perovskite_composition", "structure_pin_nip", "electron_transport_layer", "hole_transport_layer"]:
                #in top level: composition, ETL, HTL, PIN-NIP,
                curr_object[var_name] = value
            elif var_name in ["passivating_molecule", "treated_pce", "treated_voc", "control_pce", "control_voc"]:
                if len(passivators) == 1:
                    curr_object[var_name] = value
                else: ##cannot infer that this value belong to this passivator
                    if node_id in relations:
                        test_names = relations[node_id]
                        for test_name in test_names:
                            if test_name in relevant_tests:
                                curr_object[var_name] = value
            elif node_id in relations:
                test_ids = relations[node_id]
                test_names = {test_id: f"test_{test_id[-1]}" for test_id in test_ids}
                for test_id in test_ids:
                    if test_id not in relevant_tests and len(passivators) > 1: #only needs to filter by relevant tests if multiple passivators
                        continue
                    test_name = test_names[test_id]
                    if test_name in curr_object:
                        curr_object[test_name][var_name] = value
                    else:
                        curr_object[test_name] = EMPTY_STABILITY_TEST.copy()
                        curr_object[test_name][var_name] = value  
        output[passivator] = curr_object
    return output 

def parse_bioc(root):
    passivators = get_passivators(root)
    relations  = get_relations(root)
    output = get_json_for_paper(root, passivators, relations)
    return output

bioc_dir = "annotation/biocs"
data = {}
for filename in os.listdir(bioc_dir):
    if filename.endswith(".xml"):
        file_path = os.path.join(bioc_dir, filename)
        root = ET.parse(file_path).getroot()
        paper_data = parse_bioc(root)
        paper_num = extract_papernum(root)
        data[paper_num] = paper_data

with open('annotation/Schema_2/data/annotations_flattened.json', 'w') as f:
    json.dump(data, f)

with open('annotation/Schema_3_flat/data/annotations_flattened.json', 'w') as f:
    json.dump(data, f)
