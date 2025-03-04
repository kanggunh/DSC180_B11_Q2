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

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_path = model_name
print("input batch size:")  
batch_size = int(input())
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False
)

# # model can be run without quantization
# if model_index == 1:
#     bnb_config = None
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.model_max_length = 16000

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad token
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

# if model_index >= 1: #llama models only
#     pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]

PREFIX = """
"You are a scientific assistant and your task is to extract certain information from text, particularly 
in the context of perovskite solar cells. Your task is to identify and extract details about passivating molecules and associated performance data mentioned in the text.

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
        "perovskite_composition": null, // (string).
        "electron_transport_layer": null, // (string).
        "pin_nip_structure": null, // (values: "PIN" or "NIP").
        "hole_transport_layer": null, // (string).
        "passivating_molecule": null, // (string).
        "control_pce": null, // (numeric).
        "control_voc": null, // (numeric).
        "treated_pec": null, // (numeric).
        "treated_voc": null // (numeric).
        "test_1": {{ // Use unique keys for each test (e.g., test_1, test_2, etc.).
            "test_name": null, // (string)
            "temperature": null, // (numeric).
            "time": null, // (string or numeric).
            "humidity": null, // (string or numeric).
            "efficiency_cont": null, // (numeric).
            "efficiency_tret": null, // (numeric).
        }}
    }}

    Now extract from the following text:
"""
SUFFIX = """\n\n{sample}\n\n"""
def create_prompt(system, user):
    tokens = tokenizer.encode(user, max_length=8000, truncation=True) # prevents CUDA memory errors with current GPU
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

dataset.to_csv(f'data/schema2/deepseek_baseline.csv')
