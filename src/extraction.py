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
import numpy as np
from huggingface_hub import login

def run_extraction(data_path, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", batch_size=1, for_eval=False):
    """
    Runs the extraction process on the given dataset using a specified model and tokenizer.

    This function loads environment variables, initializes a language model and tokenizer with 
    specified configurations, and processes the input data to extract information using a predefined
    prompt format. The extracted data is saved to a CSV file.

    Args:
        data_path (str): Path to the CSV file containing the input data to be processed.
        model_name (str, optional): Name of the model to load. Defaults to "deepseek-ai/DeepSeek-R1-Distill-Llama-8B".
        tokenizer_name (str, optional): Name of the tokenizer to load. Defaults to "deepseek-ai/DeepSeek-R1-Distill-Llama-8B".
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 1.

    Returns:
        str: Path to the output CSV file containing the extracted JSON data.
    """

    load_dotenv()
    access_token = os.getenv("HF_TOKEN")
    login(token=access_token)

    bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=getattr(torch, "float16"),
    # bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
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

    with open("../data/prompts/fully_nested.txt", "r'") as f:
        PREFIX = f.read()
    
    def create_prompt(system, user):
        """
        Creats a prompt for the causal language model from a system and a user input.
        
        Args:
            system (str): The system prompt.
            user (str): The user input.
        
        Returns:
            list[dict]: A prompt for the causal language model.
        """
        tokens = tokenizer.encode(user, max_length=16000, truncation=True) # prevents CUDA memory errors with current GPU
        truncated_user = tokenizer.decode(tokens)
        return [
        {"role": "system", "content": system},
        {"role": "user", "content": truncated_user}, ]

    def generate_extraction_batch(texts):
        """
        Generates a batch of extracted data from a list of texts.

        Args:
            texts (list[str]): A list of texts to extract from.

        Returns:
            list[dict|str|None]: A list of extracted data, where each element is either a parsed JSON object, a raw JSON string, or None if no JSON was found.
        """
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
    
    num_workers = cpu_count()  # Adjust based on available CPUs
    dataset = pd.read_csv(data_path)
    data_loader = DataLoader(dataset["text"].tolist(), batch_size=batch_size, num_workers=num_workers)

    json_outputs = []
    for batch in tqdm(data_loader, desc="Processing Batches"):
        json_outputs.extend(generate_extraction_batch(batch))

    dataset["json_output"] = json_outputs

    file_name = data_path.split("/")[-1]
    output_path = f'../data/extraction_{"eval" if for_eval else "final"}/{file_name}'
    dataset.to_csv(output_path)
    return output_path

def escape_internal_quotes(json_string):
    # This regex finds values within quotes and fixes internal unescaped quotes
    return re.sub(r'":\s*"([^"]*?)"', lambda m: '": "' + m.group(1).replace('"', '\\"') + '"', json_string)

def force_fix(json_string):
    """
    Fixes a JSON string by replacing all double quotes with single quotes.
    This is done by first replacing all double quotes with special characters,
    then replacing the double quotes with single quotes, and then replacing
    the special characters with the original double quotes.
    """
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

def clean_json(json_string):
    """
    This function takes a JSON string and attempts to make it a valid JSON
    object. It first removes any leading/trailing whitespace, and then
    removes any "None" values and replaces them with "null". It then
    removes any single quotes and replaces them with double quotes, and
    escapes any internal double quotes. It then uses a regular expression
    to remove any colons from the keys of the JSON object. Finally, it
    removes any "treated_pec" keys and replaces them with "treated_pce".
    """
    if json_string is None: return None
    
    if "```json" in json_string:
        json_string = json_string.split("```json")[1]
        json_match = re.search(r"\{.*\}", json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(0).strip()
    json_string = json_string.replace("None", "null")
    json_string = json_string.replace("'", "\"")
    json_string = escape_internal_quotes(json_string)
    json_string = force_fix(json_string)
    json_string = re.sub(r'(".*?")', lambda m: m.group(1).replace(":", ""), json_string)
    json_string = json_string.replace("treated_pec", "treated_pce")
    return json_string

def convert_csv_to_json(csv_path, output_path):
    """
    Reads a CSV file at `csv_path`, extracts the column "json_output", processes it
    into a valid JSON string and writes it to a JSON file at `output_path`.

    The function processes the JSON string as follows:
        - If the string is None or NaN, it is skipped.
        - It searches for the first occurrence of a JSON object in the string.
        - If such an object is found, it is cleaned and then parsed into a JSON object.
        - If parsing fails, the cleaned string is written to the output.
        - If no JSON object is found, the string is written to the output as is.
    """
    df = pd.read_csv(csv_path)
    output = {}
    i = 0
    for _, row in df.iterrows():
        i += 1
        json_output = row["json_output"]
        if json_output is None or json_output is np.nan:
            output[str(row["id"])] = None
            continue
        json_match = re.search(r"\{.*\}", json_output, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0).strip()
            try:
                cleaned_json = json.loads(clean_json(raw_json))
            except json.JSONDecodeError as e:
                output[str(row["id"])] = clean_json(raw_json)
        else:
            output[str(row["id"])] = json_output
    with open(output_path, 'w') as f:
        json.dump(output, f)