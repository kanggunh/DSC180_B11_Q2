from openai import OpenAI
import openai
import json
import pandas as pd
from dotenv import load_dotenv
import os
import re
import requests



def get_system_content():
    """
    Returns a string that is a docstring for an OpenAI function.
    The string is a docstring for a function that takes a JSON object as a string and formats the JSON object according to a certain schema.
    The schema is as follows:
    {
    "perovskite_composition": {"type:": str},
    "electron_transport_layer": {"type": str},
    "hole_transport_layer": {"type": str},
    "structure_pin_nip": {"type": str},
    "passivating_molecule": {"type": str},
    "control_pce": {"type": float},
    "treated_pce": {"type": float},
    "control_voc": {"type": float},
    "treated_voc": {"type": float},
    "test_1": {
        "stability_type": {"type": str},
        "humidity": {"type": int},
        "temperature": {"type": int},
        "time": {"type": int},
        "retained_proportion_cont": {"type": float},
        "retained_proportion_tret": {"type": float}
        }
    }
    The function is meant to be used with the OpenAI API, and is meant to be used to format JSON objects that are passed to it.
    """
    json_schema = {
    "perovskite_composition": {"type:": str},
    "electron_transport_layer": {"type": str},
    "hole_transport_layer": {"type": str},
    "structure_pin_nip": {"type": str},
    "passivating_molecule": {"type": str},
    "control_pce": {"type": float},
    "treated_pce": {"type": float},
    "control_voc": {"type": float},
    "treated_voc": {"type": float},
    "test_1": {
        "stability_type": {"type": str},
        "humidity": {"type": int},
        "temperature": {"type": int},
        "time": {"type": int},
        "retained_proportion_cont": {"type": float},
        "retained_proportion_tret": {"type": float}
        }
    }
    ensure_format_system = """
    You are a helpful data quality assistant.
    Your task is to make sure that each value of the keys in the JSON object provided follows the following schema:

    {json_schema}

    **Instructions**
    - Note that there may be multiple tests, where each test has the key (test_1, test_2, etc.)
    The JSON provided to you may not follow this exact format. Rename the keys if necessary to ensure that it follows the schema previously outlined.
    - The JSON provided may have values that cannot be parsed into JSON (e.g. "temperature": 85 C). For these values, simply put the value as a string.
    - For the following variables, make sure the units are correct. Make conversions if necessary. There are some situations where the value is an equation, (e.g. 18 * 24). In these cases, solve the equation and return the value as a number.
        - "time": hours (as an int)
        - "humidity": percent (as an int)
        - "temperature": celsius (as an int)
    - Once the values are converted to the proper unit, drop the units from the value, so that it can be a number.
    - The JSON provided may be missing some keys. In the case a key is missing, put it in the JSON with its value as null.
    - Return the original JSON object but where each key's value is the JSON object you formatted.
    - Never drop a key value pair from the original JSON object.
    - If any value is a list, return the first value in the list.
    - Make sure to format every JSON object given to you.
    - Ensure that the value you return can be parsed using json.loads().

    Begin formatting!
    """
    return ensure_format_system.format(json_schema=json_schema)

def ensure_json_format(json_path, batch_size=50):
    """
    Takes a path to a JSON file and formats all the objects in the JSON according to the schema provided in get_system_content().

    The function takes the path to a JSON file and a batch size as arguments. The batch size is the number of objects to format at once. The function will go through each object in the JSON file and format it according to the schema provided in get_system_content(). The function will return the path to a new JSON file that contains the formatted objects.

    Uses the OpenAI API to format the objects. The function is meant to be used to format JSON objects that are passed to it.

    Returns a path to a new JSON file that contains the formatted objects. The function will not modify the original JSON file.

    Prints out the keys of the objects it is processing, and will print out any objects that it was unable to format.

    Returns the path to the new JSON file as a string.

    Parameters
    ----------
    json_path : str
        The path to the JSON file.
    batch_size : int, optional
        The number of objects to format at once. The default is 50.

    Returns
    -------
    str
        The path to the formatted JSON file.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    system_content = get_system_content()
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    keys = list(data.keys())
    num_objects = len(keys)
    l = 0
    formatted_objects = {}
    while l < num_objects:
        r = l + batch_size
        if r > num_objects:
            r = num_objects
        curr_keys = keys[l:r]
        print("processing keys: ", curr_keys)
        curr_batch = {key: data[key] for key in curr_keys if key in data}
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": str(curr_batch) 
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=False,
            max_tokens=16384
        )
        output = response.choices[0].message.content
        json_match = re.search(r"\{.*}", output, re.DOTALL)
        if not json_match:
            print(f"could not format keys {curr_keys[l]} to {curr_keys[r]}")
            formatted_objects.update(curr_batch)
        else:
            result = json_match.group(0).strip()
            print(result)
            output_json = json.loads(result)
            formatted_objects.update(output_json)
        l = r
    file_name, file_extension = os.path.splitext(json_path)
    formatted_file_path = file_name + "_formatted" + file_extension
    with open(formatted_file_path, "w") as f:
        json.dump(formatted_objects, f)
    return formatted_file_path

def get_new_passivator_conversions(passivators, api_key_name="OPENAI_API_KEY", base_url="https://api.openai.com/v1/", model_name="gpt-4o"):    
    PREFIX = """
    Role:
    You are a helpful scientific assistant specializing in passivating molecules for perovskite solar cells.

    Task:
    Given a list of molecule names, provide the full name of each molecule in IUPAC nomenclature. The IUPAC name must be parseable into SMILES format.

    Rules:
    -If the molecule name is already in IUPAC format, return it as is.
    -If there are multiple molecule names given, just use the first one.
    -If the name contains additional descriptive words (e.g., "passivating," "functionalized"), extract only the molecule name and convert it to IUPAC format.
    -If the full molecule name cannot be parsed into SMILES after fully reasoning through it multiple times, return None.
    -Ensure no molecules are left out.

    Output Format:
    Provide a JSON object where each key is the provided molecule name and the value is the corresponding IUPAC name or None.

    json
    Copy
    {
        "molecule_name_1": "IUPAC_name_or_None",
        "molecule_name_2": "IUPAC_name_or_None"
    }
    Important Notes:

    -Focus only on the parts of the string that represent the molecule name.

    -Double-check that the IUPAC name can be parsed into SMILES. If not, return None.
    -If the molecule is not relevant to passivating perovskite solar cells, return None.

    Example Input:

    {
        "ethylammonium bromide": "ethylammonium bromide",
        "passivating 2-phenylethylamine": "2-phenylethylamine",
        "CF3PEAI": "2-(4-(Trifluoromethyl)phenyl)ethylammonium iodide",
        "2D perovskite": None
    }
    Begin converting!
    """
    api_key = os.getenv(api_key_name)
    client = OpenAI(api_key=api_key, base_url=base_url)
    system_content = PREFIX
    i = 0
    user_content = str(passivators)

    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False
    )
    output = response.choices[0].message.content
    json_match = re.search(r"\{.*\}", output, re.DOTALL)

    if json_match:
        raw_json = json_match.group(0).strip()
    else:
        print("No JSON found")
        return {}
    try:
        parsed_data = json.loads(raw_json)
        return parsed_data
    except json.JSONDecodeError as e:
        print("Error creating JSON", e)
        return {}



def get_unconverted_passivators(extraction_data, passivator_conversions):
    """
    Extracts all passivators from extraction data that have not been converted to standardized names yet.

    Args:
        extraction_data (dict): A dictionary containing the extraction data where the passivators are to be found.
        passivator_conversions (dict): A dictionary containing the passivator names as keys and their standardized names as values.

    Returns:
        list: A list of passivators that have not been converted to standardized names yet.
    """
    passivators = set()
    search_key = "passivating_molecule"
    for key in extraction_data:
        item = extraction_data[key]
        for paper_key in item:
            if paper_key == search_key:
                passivators.add(item[paper_key])
    passivators = [passivator for passivator in passivators if passivator not in passivator_conversions and passivator not in [None, 'none']]
    return passivators    
        
def format_passivators(json_path):
    """
    Takes a JSON file and formats the passivating molecules within it.
    First, it goes through the file and finds all the passivating molecules that have not been converted to standardized names yet.
    It then uses the GPT-4o model to convert these passivating molecules to standardized names.
    It makes 3 passes to ensure that it is not missing anything.
    After it is done, it updates the passivator_conversions.json file with the new conversions and then goes through the JSON file and updates all the passivating molecules to their standardized names.
    """
    passivator_conversions_path = '../data/extraction_final/passivator_conversions.json'
    with open(passivator_conversions_path, 'r') as json_file:
        passivator_conversions = json.load(json_file)
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    passivators = get_unconverted_passivators(data, passivator_conversions)
    new_passivators = get_new_passivator_conversions(passivators)
    ### second pass to ensure not missing anything (GPT-4o takes less time for this)
    None_passivators = [k for k, v in new_passivators.items() if v == None or v == "None"]
    extracted_passivators = {k: v for k, v in new_passivators.items() if v != None or v != "None"}

    second_pass = get_new_passivator_conversions(None_passivators)
    extracted_passivators.update(second_pass)

    None_passivators = [k for k, v in extracted_passivators.items() if v == None or v == "None"]
    extracted_passivators = {k: v for k, v in extracted_passivators.items() if v != None or v != "None"}

    third_pass = get_new_passivator_conversions(None_passivators)
    extracted_passivators.update(third_pass)

    passivator_conversions.update(extracted_passivators)

    with open(passivator_conversions_path, 'w') as json_file:
        json.dump(passivator_conversions, json_file)

    for key in data:
        if "passivating_molecule" in data[key]:
            passivator_key = data[key]["passivating_molecule"]
            if passivator_key in passivator_conversions:
                data[key]["passivating_molecule"] = passivator_conversions[passivator_key]
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

def get_smiles(passivating_molecule):
    """
    Takes a passivating molecule and returns its SMILES string if it exists, None otherwise.

    Args:
        passivating_molecule (str): The name of the passivating molecule.

    Returns:
        str or None: The SMILES string of the passivating molecule if it exists, None otherwise.
    """
    base_url = "https://opsin.ch.cam.ac.uk/opsin/"
    smiles_url = base_url + passivating_molecule + ".smi"
    r = requests.get(smiles_url)
    return r.text if r.status_code == 200 else None

def create_df_from_json(json_path):
    """
    Takes a JSON file from extraction output and converts it into a DataFrame. The DataFrame is then sorted by index.
    It then adds a new column "passivator_smiles" which is the SMILES string for the passivating molecule if it exists.
    The DataFrame is then saved to a CSV file.
    Returns the path to the output CSV file.
    """
    df = pd.read_json(json_path)
    df = df.T.sort_index()
    df["passivator_smiles"] = df["passivating_molecule"].apply(get_smiles)
    output_path = "../data/extraction_final/data_with_smiles.csv"
    df.to_csv(output_path, index=False)
    return output_path