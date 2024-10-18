import os
import json
import argparse
import re
from src.utils1 import QADataset, locate_answer, locate_answer4pub_llama
from sklearn.metrics import accuracy_score
import numpy as np
import statistics



def evaluate(dataset, save_dir, split="test", locate_fun=locate_answer):

    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans:i for i, ans in enumerate(answer_list)}
    
    total_len = len(dataset)

    # for i, fpath in enumerate(sorted([f for f in os.listdir(save_dir) if f.endswith(".json")])[:total_len]):
    for q_idx in range(100):
        fpath = os.path.join(save_dir, split + "_" + dataset.index[q_idx] + ".json")
        answers = []
        for it in json.load(open(fpath))[:1]:
            answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        # answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        answers = [ans for ans in answers if ans != "NA"]
        if len(answers) == 0:
            pred.append(-1)
            continue
        ans = statistics.mode(answers)
        if ans in answer_list:
            pred.append(answer_list.index(ans))
        else:
            pred.append(-1)
    
    truth = [answer2idx[item['answer']] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True
    
    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1-acc) / len(truth))
    return acc, std, flag

if __name__ == "__main__":

    dataset_names = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq']
    datasets = {key:QADataset(key) for key in dataset_names}

    results_dir="./prediction"
    llm_name="axiong/PMC_LLaMA_13B"

    scores = []
    rag=False
    corpus_name=""
    retriever_name=""

    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="")
        split = "test"
        if dataset_name == "medmcqa":
            split = "dev"
        if rag:
            save_dir = os.path.join(results_dir, dataset_name, "rag_"+str(k), llm_name, corpus_name, retriever_name)
        else:
            save_dir = os.path.join(results_dir, dataset_name, "cot", llm_name)
            print(save_dir)
        if os.path.exists(save_dir):
            if "pmc_llama" in llm_name.lower():
                acc, std, flag = evaluate(datasets[dataset_name], save_dir, split, locate_answer4pub_llama)
            else:
                acc, std, flag = evaluate(datasets[dataset_name], save_dir, split)
            scores.append(acc)
            print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
            if flag:
                print(" (NOT COMPLETED)")
            else:
                print("")
        else:
            print("NOT STARTED.")
            # scores.append(0)

    if len(scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))