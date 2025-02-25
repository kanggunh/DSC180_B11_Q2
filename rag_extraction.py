from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import torch
import pandas as pd
from huggingface_hub import login

load_dotenv()
access_token = os.getenv("HF_TOKEN")
login(token=access_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=getattr(torch, "float16"),
    # bnb_4bit_use_double_quant=False
)

# trained model path
model_name = "DSC180_B11_Q2/models/DeepSeek-R1-PSC-Extractor-8B-8bit"
# model_path = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.model_max_length = 70000
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=None,
    top_p=None,
    do_sample=False,
)

# prompt that specifies what to return
PREFIX = """
"You are a scientific assistant and your task is to extract certain information from text, particularly 
in the context of perovskite solar cells. Your task is to identify and extract details about passivating molecules and associated performance data mentioned in the text.
We are in a scientific environment. You MUST be critical of the units of the variables.

"Only extract the variables that were developed in this study. You must omit the ones extracted from the bibliography"
Your task is to extract relevant scientific data from the provided text about perovskite solar cells.
    Follow these guidelines:

    1. **If passivating molecules are mentioned:**
    - Include stability test data for each molecule if available. There may be multiple stability tests for a single molecule.

    2. **If no passivating molecules are mentioned:**
    - Provide a JSON object with any other relevant data explicitly mentioned in the text.

    **JSON Structure:**
    - DO NOT change the names of any of the property names. It is imperative that these are exactly as they as stated in the schema below.
    - Ensure the output adheres to the following structure and is parseable as valid JSON:

    {{
        "perovskite_composition": null, // Chemical formula of the perovskite (string).
        "electron_transport_layer": null, // Material used as the electron transport layer (string).
        "pin_nip_structure": null, // Whether the perovskite uses a PIN or NIP structure (values: "PIN" or "NIP").
        "hole_transport_layer": null, // Material used as the hole transport layer (string).
        "test_1": {{ // Include only if stability tests are mentioned. Use unique keys for each test (e.g., test_1, test_2, etc.).
            "test_name": null, // Must be one of: "ISOS-D", "ISOS-L", "ISOS-T", "ISOS-LC", "ISOS-LT".
            "temperature": null, // Temperature in Celsius (numeric or string, no units or symbols like ° or -).
            "time": null, // Duration of the test in hours (string or numeric).
            "humidity": null, // Humidity level (string or numeric).
            "retained_percentage_cont": null, // Percentage of the PCE retained by the control perovskite after stability test (numeric) (values should be between 30-100).
            "retained_percentage_tret": null, // Percentage of the PCE retained by the treated perovskite after stability test (numeric) (values should be between 30-100).
            "passivating_molecule": null, // Name of the passivating molecule used in the test (must be a proper molecule name - i.e. can be parsed into SMILES format).
            "control_pce": null, // Power conversion efficiency for control perovskite (numeric) (values should be between 10-30).
            "control_voc": null, // Open-circuit voltage for control perovskite (numeric).
            "treated_pce": null, // Power conversion efficiency for treated perovskite (numeric) (values should be between 10-30).
            "treated_voc": null // Open-circuit voltage for treated perovskite (numeric).
        }}
    }}

    **Instructions:**
    - Be concise and accurate. Include only data explicitly present in the text.
    - For stability tests:
    - Infer the test type (e.g., ISOS-D, ISOS-L) based on the description if not explicitly stated.
    - Ensure all numeric values are parseable (e.g., no symbols like ° or -).
    - Use unique keys for each test (e.g., `test_1`, `test_2`, etc.).
    - If a field has no data, set it to `null`.
    - The data may be mentioned in units different from the ones specified in the schema. In this case, convert it into the desired unit (e.g. 30 days becomes 720 hours)
    - Make sure to only return a JSON object.
    - Do not create any properties that are not stated in the JSON structure provided.
    - If you cannot find a value, do not omit that property, just set it to null.
    - Make sure not to confuse the retained_proportion_cont/retained_proportion_tret variables with the control_pce/treated_pce variables. 
    - The PCE values will almost never be above 30, while the percentage retained values will rarely be below 50%. The retained percentage will not always be there, 
    please leave these values as null if they cannot be found. DO NOT use the PCE for these values.

    Now extract from the following text:
"""
SUFFIX = """\n\n{sample}\n\n"""
def create_prompt(system, user):
    tokens = tokenizer.encode(user, max_length=60000, truncation=True) # prevents CUDA memory errors with current GPU
    truncated_user = tokenizer.decode(tokens)
    return [
    {"role": "system", "content": system},
    {"role": "user", "content": truncated_user}, ]

def generate_extraction(text):
    instruction = create_prompt(PREFIX, text)
    json_string = pipe(instruction, max_new_tokens=4096)[0]["generated_text"][-1]['content']
    json_match = re.search(r"\{.*\}", json_string, re.DOTALL)
    if json_match:
        raw_json = json_match.group(0).strip()
    else:
        print("No JSON found")
        return json_string
    # Fix unquoted values (if any exist, less common)
    parseable = re.sub(r'":\s*"([^"]*?)(".*?"[^"]*?)"', r'": "\1\\\2"', raw_json)
    try:
        parsed_data = json.loads(parseable)
        return parsed_data
    except json.JSONDecodeError as e:
        print("Error creating JSON", e)
        return parseable
    
dataset = pd.read_csv('DSC180_B11_Q2/data/rag_filtered_150_papers.csv')
dataset["json_output"] = dataset["filtered_text"].apply(generate_extraction)
# output = {}
# for index, row in dataset:
#     output[str(row["id"])] = row["json_output"]
# with open('DSC180_B11_Q2/data/deepseek_8bit_finetuned.json', 'w') as f:
#     json.dump(output, f)

dataset.to_csv('DSC180_B11_Q2/data/deepseek_8bit_finetuned_flattened.csv')