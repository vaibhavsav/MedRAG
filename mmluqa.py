from src.medrag import MedRAG
from datasets import load_dataset
import re
import torch
import time
import gc

from src.utils1 import QADataset,locate_answer,locate_answer4pub_llama
# from torch.utils.data import Dataset, DataLoader


# from multiprocessing import Process,set_start_method
llama_3_1_8B="meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_3_2_1B="meta-llama/Llama-3.2-1B-Instruct"
pmc_llama = 'axiong/PMC_LLaMA_13B'

medrag = MedRAG(llm_name=pmc_llama, rag=False, retriever_name="Contriever", corpus_name="StatPearls")

desired_subjects = ['anatomy', 'clinical_knowledge', 'professional_medicine', 'human_genetics', 'college_medicine', 'college_biology']
dataset = load_dataset('cais/mmlu', "all")

def filter_and_convert_to_pandas(ds, desired_subjects):
        # Filter the dataset
        filtered_ds = ds.filter(lambda row: row['subject'] in desired_subjects)
        # Convert the filtered dataset to Pandas DataFrame
        return filtered_ds.to_pandas()

filtered_datasets = {}


# def process_batch(batch_df):
#     count = 0
#     for index, row in batch_df.iterrows():
#         question = row['question']  # Adjust column name as necessary
#         options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
#         torch.cuda.empty_cache()
#         gc.collect()
#         answer, _, _ = medrag.answer(question=question, options=options, k=8)
#         choice = locate_answer(answer)
#         value = row['answer']
#         if choice == str(str(chr(65 + value))):
#             count += 1
#         print(f'Extracted answer: {choice} and actual answer: {str(chr(65 + value))}')
#         time.sleep(1)
#     print(f'Batch processed. Correct answers: {count}')

# if __name__ == "__main__":

#     try:
#             set_start_method('spawn', force=True)  # Ensure 'spawn' is used
#     except RuntimeError:
#             pass
#         # Split data into batches

#     for split in dataset:
#         print(f"Processing {split} split...")
#         # Filter and convert each split to a DataFrame
#         filtered_datasets[split] = filter_and_convert_to_pandas(dataset[split], desired_subjects)

# # Now `filtered_datasets` is a dictionary where each key (split) has a filtered Pandas DataFrame
# # For example, to get the DataFrame for the "test" split:
#     filtered_test_df = filtered_datasets['test']

#     batch_size = 150
#     batches = [filtered_test_df.iloc[i:i+batch_size] for i in range(0, len(filtered_test_df), batch_size)]

#     # Process each batch in separate processes
#     for batch_df in batches:
#         p = Process(target=process_batch, args=(batch_df,))
#         #,))
#     # ,))
#         p.start()
#         p.join()  # Wait for the process to finish before starting the next one
#     #``
from multiprocessing import Process, set_start_method

def process_batch(batch_df, device_id):
    torch.cuda.set_device(device_id)
    count = 0
    for index, row in batch_df.iterrows():
        question = row['question']  # Adjust column name as necessary
        options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
        torch.cuda.empty_cache()
        gc.collect()
        answer, snippets, scores = medrag.answer(question=question, options=options, k=32)
        print(answer)
        choice = locate_answer4pub_llama(answer)
        value = row['answer']
        if choice == str(chr(65 + value)):
            count += 1
        print(f'Index: {index}, Extracted answer: {choice},  actual answer: {str(chr(65 + value))} , score: {scores}')
        time.sleep(1)
    print(f'Batch processed on device {device_id}. Correct answers: {count}')


def process_batches(batches, device_id):
    import torch
    torch.cuda.set_device(device_id)
    # Initialize medrag or any other necessary resources here
    for batch_df in batches:
        process_batch(batch_df, device_id)

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)  # Ensure 'spawn' is used for multiprocessing
    except RuntimeError:
        pass

    for split in dataset:
        print(f"Processing {split} split...")
        # Filter and convert each split to a DataFrame
        filtered_datasets[split] = filter_and_convert_to_pandas(dataset[split], desired_subjects)

    filtered_test_df = filtered_datasets['test']
    # Assuming 'filtered_test_df' is your DataFrame and 'batch_size' is defined
    batch_size = 150
    batches = [filtered_test_df.iloc[i:i+batch_size] for i in range(0, len(filtered_test_df), batch_size)]

    device_ids = [0]  # IDs of your GPUs
    processes = []
    timeStart = time.time()
    for i, device_id in enumerate(device_ids):
        # Assign batches to this device: every nth batch starting from i
        device_batches = batches[i::len(device_ids)]
        p = Process(target=process_batches, args=(device_batches, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    timeEnd = time.time()
    print(timeEnd-timeStart)
