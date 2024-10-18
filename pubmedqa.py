import os
import json
import argparse
import re
from src.utils1 import QADataset, locate_answer, locate_answer4pub_llama
from sklearn.metrics import accuracy_score
import numpy as np
import statistics
from src.medrag import MedRAG
import torch
import gc


pmc_llama = 'axiong/PMC_LLaMA_13B'
medrag = MedRAG(llm_name=pmc_llama, rag=False)


# Function to extract questions, options, and answers
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
    medrag = MedRAG(llm_name=pmc_llama, rag=False)
    with open('benchmark.json', 'r') as file:
        data = json.load(file)
    
    medqa = data['pubmedqa']

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
        print(count)
    
    print(f"The number of correct answers: {count}")
    

    