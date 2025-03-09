from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from huggingface_hub import login

def create_prompt_formats(sample):
    """
    Formats a given prompt sample by adding the instructions and intro to the top, 
    and the response to the bottom, and returns the formatted sample.
    
    Parameters:
    sample (dict): A dictionary containing the keys 'text', 'output', and 'memory'.
    
    Returns:
    dict: The formatted sample with the instructions and intro added to the top, 
    and the response to the bottom.
    """
    INTRO_BLURB = (
    "You are a scientific assistant and your task is to extract certain information from text, particularly"
    "in the context of perovskite solar cells. Your task is to identify and extract details about passivating molecules and associated performance data mentioned in the text."
    "We are in a scientific environment. You MUST be critical of the units of the variables."
    "Only extract the variables that were developed in this study. You must omit the ones extracted from the bibliography"
    )
    INSTRUCTION_KEY = """### Instruct: 
    Your task is to extract relevant scientific data from the provided text about perovskite solar cells. Then join them with the data from previous chunks.
    It is likely that a lot of this data is not present in the chunk provided. Only extract the data points that are present in the chunk.
    Follow these guidelines:

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
        "treated_pce": null, // Power conversion efficiency for treated perovskite (numeric) (values should be between 10-30).
        "treated_voc": null // Open-circuit voltage for treated perovskite (numeric).
        "test_1": {{ // Include only if stability tests are mentioned. Use unique keys for each test (e.g., test_1, test_2, etc.).
            "test_name": null, // Must be one of: "ISOS-D", "ISOS-L", "ISOS-T", "ISOS-LC", "ISOS-LT".
            "temperature": null, // Temperature in Celsius (numeric or string, no units or symbols like ° or -).
            "time": null, // Duration of the test in hours (string or numeric).
            "humidity": null, // Humidity level (string or numeric).
            "efficiency_cont": null, // Percentage of the PCE retained by the control perovskite after stability test (numeric) (values should be between 30-100).
            "efficiency_tret": null, // Percentage of the PCE retained by the treated perovskite after stability test (numeric) (values should be between 30-100).
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
    - Make sure not to confuse the efficiency_cont/efficiency_tret variables with the control_pce/treated_pce variables. 
    - The PCE values will almost never be above 30, while the percentage retained values will rarely be below 50%. The retained percentage will not always be there, 
    please leave these values as null if they cannot be found. DO NOT use the PCE for these values.

    Text to extract from:

    {chunk}

    Add the newly extracted data to the one from previous chunks that is the following:

    {memory}

    Never leave the information from previous chunks behind.
    """
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"


    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY.format(chunk=sample['text'], memory=sample['memory'])}"
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

def finetune_extraction_model(model_name, training_data_path, batch_size=2, num_epochs=3):
    """
    Finetunes a given extraction model on annotated and chunked dataset.

    Args:
    model_name (str): The name of the model to finetune.
    training_data_path (str): The path to the training data csv file.
    batch_size (int, optional): Defaults to 2.
    num_epochs (int, optional): Defaults to 3.

    Returns:
    str: The path to the saved model.
    """
    load_dotenv()
    df = pd.read_csv(training_data_path)
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    access_token = os.getenv("HF_TOKEN")
    login(token=access_token)
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    device_map = "auto"
    original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        quantization_config=bnb_config,
                                                        device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    from functools import partial

    def preprocess_batch(batch, tokenizer, max_length):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True
        )

    def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset):

        dataset = dataset.map(create_prompt_formats)

        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["text","memory", "output"]
        )

        dataset = dataset.shuffle()

        return dataset
    MAX_LENGTH = 2000

    train_dataset = preprocess_dataset(tokenizer, MAX_LENGTH, dataset['train'])
    eval_dataset = preprocess_dataset(tokenizer, MAX_LENGTH, dataset['test'])
    

    config = LoraConfig(
        r=32, #Rank
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    original_model = prepare_model_for_kbit_training(original_model)
    original_model.gradient_checkpointing_enable()
    # original_model = torch.nn.DataParallel(original_model, device_ids=[0,1,2,3])

    peft_model = get_peft_model(original_model, config)
    output_dir = f'models/peft-dialogue-summary-training-{str(int(time.time()))}'
    import transformers

    peft_training_args = TrainingArguments(
        learning_rate=6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=batch_size,
        optim="paged_adamw_8bit",
        num_train_epochs=num_epochs,
        fp16=False,
        bf16=False,  # bf16 to True with an A100, False otherwise
        logging_steps=5,  # Logging is done every step.
        evaluation_strategy="steps",
        max_grad_norm=0.3,
        warmup_steps=100,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        output_dir="../results/",
        logging_dir="logs",
        save_strategy="steps",
        eval_steps=5,
        do_eval=True,
        save_steps=400,
        save_total_limit=15,
        report_to="none"
    )

    peft_model.config.use_cache = False

    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    print("training about to begin")
    peft_trainer.train()
    try:
        model_path = f"../models/PSC-Extractor-{model_name.split('/')[-1]}"
        peft_trainer.model.save_pretrained(model_path)
    except:
        print("failed to save to original path")
    return model_path
