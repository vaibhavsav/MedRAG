from src.medrag import MedRAG
from datasets import load_dataset
import re
import torch
import time
import gc
import json

from src.utils1 import QADataset,locate_answer,locate_answer4pub_llama
# from torch.utils.data import Dataset, DataLoader


# from multiprocessing import Process,set_start_method
llama_3_1_8B="meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_3_2_1B="meta-llama/Llama-3.2-1B-Instruct"
pmc_llama = 'axiong/PMC_LLaMA_13B'

medrag = MedRAG(llm_name=pmc_llama, rag=True, retriever_name="RRF-4", corpus_name="MedCorp")

def extract_data(data):
    extracted_data = []
    
    for key, value in data.items():
        question = value.get('question')
        options = value.get('options')
        answer = value.get('answer')
        
        extracted_data.append({
            'question': question,
            'options': options,
            'answer': answer
        })
    
    return extracted_data

pmc_llama = 'axiong/PMC_LLaMA_13B'

if __name__ == "__main__":
    #medrag = MedRAG(llm_name=pmc_llama, rag=True)
    with open('benchmark.json', 'r') as file:
        data = json.load(file)
    
    medqa = data['mmlu']

    extracted_data = extract_data(medqa)
    count = 0
    for item in extracted_data:
        question = item['question']
        options = item['options'] 
        torch.cuda.empty_cache()
        gc.collect()
        answer, snippets, scores = medrag.answer(question=question, options=options, k=32)
        print(answer)
        choice = locate_answer4pub_llama(answer)
        value = item['answer']
        if choice == str(item['answer']):
            count += 1
        print(f"Expected Answer: {choice} , Actual Answer: {value}")
    
    print(f"The number of correct answers: {count}")
    

    
