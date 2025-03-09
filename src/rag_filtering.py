from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import pandas as pd

def filter_with_rag(data_path):
    """
    This function takes a path to a csv file containing text data and
    uses the RAG model to filter the text into relevant chunks based on
    a simple prompt. The filtered text is then saved to a new csv file
    with the same filename as the original file but with "_rag_filtered"
    appended to the beginning of the filename.

    Parameters
    ----------
    data_path : str
        The path to the csv file containing the text data.

    Returns
    -------
    output_path : str
        The path to the new csv file containing the RAG filtered text.
    """
    data = pd.read_csv(data_path)
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    simple_prompt = "What, if any, is the passivating molecule tested, and what is the corresponding PCE, VOC, and stability test data (efficiency retained over time, temperature, test type)"

    query = model.encode(simple_prompt, convert_to_tensor=True)

    rag_filtered_data = []
    for index, row in data.iterrows():
        curr_text = row["text"]
        chunks = curr_text.split('\n')
        chunk_num = 1 #skips paper id
        new_chunks = []
        while chunk_num < len(chunks):
            chunk = chunks[chunk_num]
            chunk_num += 1
            if chunk.startswith('\t\t\t') or len(chunk.strip()) == 0:
                continue
            new_chunks.append(chunk)
        print(new_chunks[0])
        cosine_similarities = []
        for chunk in new_chunks:
            text_embeddings = model.encode(chunk, convert_to_tensor=True)
            cosine_similarities.append(util.pytorch_cos_sim(query, text_embeddings).item())

        cos_mean = np.mean(cosine_similarities)

        classified_chunks = []
        for i, value in enumerate(cosine_similarities):
            if value >= cos_mean:
                classified_chunks.append(new_chunks[i])
        row["filtered_text"] = '\n'.join(classified_chunks)
        rag_filtered_data.append(row)
    rag_df = pd.DataFrame(rag_filtered_data)
    filename = os.path.basename(data_path).split('.')[0]
    output_path = f'../rag_processing/rag_filtered_{filename}.csv'
    rag_df.to_csv(output_path, index=False)
    return output_path