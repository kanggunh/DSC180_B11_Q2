from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import pandas as pd
from huggingface_hub import login

load_dotenv()
access_token = os.getenv("HF_TOKEN")
login(token=access_token)

model_names = ["DeepSeek-R1-PSC-Extractor-8B-8bit-Schema-2", "Llama-PSC-Extractor-3B-16bit", "LLama-PSC-Extractor-8B-8bit-Schema-2"]
print("input model name:")
model_name = input()
if model_name not in model_names:
    print("invalid model name")
    exit()
model_index = model_names.index(model_name)
tokenizers = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
tokenizer_name = tokenizers[model_index]
model_path = os.path.join("models", model_name)
print("input batch size:")  
batch_size = int(input())
print("")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=getattr(torch, "float16"),
    # bnb_4bit_use_double_quant=False
)

# model can be run without quantization
if model_index == 1:
    bnb_config = None
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.model_max_length = 70000
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = model.config.eos_token_id
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=None,
    top_p=None,
    do_sample=False,
)

pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

PREFIX = """
"You are a scientific assistant and your task is to extract certain information from text, particularly 
in the context of perovskite solar cells. Your task is to identify and extract details about passivating molecules and associated performance data mentioned in the text.
We are in a scientific environment. You MUST be critical of the units of the variables.

"Only extract the variables that were developed in this study. You must omit the ones extracted from the bibliography"
Your task is to extract relevant scientific data from the provided text about perovskite solar cells.
    Follow these guidelines:

    1. **If passivating molecules are mentioned:**
    - If there is more than one passivating molecule tested, only return data for the champion passivator.
    - Include stability test data for the champion passivating molecule. There may be multiple stability tests for a single molecule.

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
        "passivating_molecule": null, // Name of the passivating molecule used in the test (must be a proper molecule name - i.e. can be parsed into SMILES format).
        "control_pce": null, // Power conversion efficiency for control perovskite (numeric) (values should be between 10-30).
        "control_voc": null, // Open-circuit voltage for control perovskite (numeric).
        "treated_pec": null, // Power conversion efficiency for treated perovskite (numeric) (values should be between 10-30).
        "treated_voc": null // Open-circuit voltage for treated perovskite (numeric).
        "test_1": {{ // Include only if stability tests are mentioned. Use unique keys for each test (e.g., test_1, test_2, etc.).
            "test_name": null, // Must be one of: "ISOS-D", "ISOS-L", "ISOS-T", "ISOS-LC", "ISOS-LT".
            "temperature": null, // Temperature in Celsius (numeric or string, no units or symbols like ° or -).
            "time": null, // Duration of the test in hours (string or numeric).
            "humidity": null, // Humidity level (string or numeric).
            "retained_percentage_cont": null, // Percentage of the PCE retained by the control perovskite after stability test (numeric) (values should be between 30-100).
            "retained_percentage_tret": null, // Percentage of the PCE retained by the treated perovskite after stability test (numeric) (values should be between 30-100).
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

def generate_extraction_batch(texts):
    instructions = [create_prompt(PREFIX, text) for text in texts]
    results = pipe(instructions, max_new_tokens=4096, batch_size=4)  # Increase batch size as needed
    extracted_jsons = []
    
    for res in results:
        json_string = res[0]["generated_text"][-1]['content']
        json_match = re.search(r"\{.*\}", json_string, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0).strip()
        else:
            extracted_jsons.append(None)
            continue

        try:
            parsed_data = json.loads(raw_json)
            extracted_jsons.append(parsed_data)
        except json.JSONDecodeError:
            extracted_jsons.append(raw_json)

    return extracted_jsons

num_workers = min(cpu_count(), 8)  # Adjust based on available CPUs
dataset = pd.read_csv('data/rag_filtered_150_papers.csv')
data_loader = DataLoader(dataset["filtered_text"].tolist(), batch_size=batch_size, num_workers=num_workers)

json_outputs = []
for batch in tqdm(data_loader, desc="Processing Batches"):
    json_outputs.extend(generate_extraction_batch(batch))

dataset["json_output"] = json_outputs
# output = {}
# for index, row in dataset:s
#     output[str(row["id"])] = row["json_output"]
# with open('DSC180_B11_Q2/data/deepseek_8bit_finetuned.json', 'w') as f:
#     json.dump(output, f)

dataset.to_csv(f'data/schema2/{model_name}.csv')
