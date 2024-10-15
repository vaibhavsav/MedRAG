from src.parallel_medrag_gpu import MedRAG
from datasets import load_dataset
import re
import torch
import time
import gc

from src.utils1 import QADataset,locate_answer
# from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

# question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
# options = {
#     "A": "paralysis of the facial muscles.",
#     "B": "paralysis of the facial muscles and loss of taste.",
#     "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
#     "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
# }





# dataset_name = "mmlu"
# dataset = QADataset(dataset_name)





# print(len(dataset))
# 1089

# print(dataset[0])

# import re
# import concurrent.futures

# def process_row(index, row):
#     question = row['question']
#     options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
    
#     # Get the answer from medrag
#     answer, snippets, scores = medrag.answer(question=question, options=options, k=32)

#     # Pattern to extract answer
#     pattern = r'(?i)(answer[_ ]?choice|best answer is|correct answer is)\W?["\']?\s*([A-D])["\']?'
    
#     # Extract the answer choice
#     match = re.search(pattern, answer)
#     if match:
#         choice = {match.group(2)}
#         print(f'Extracted answer: {choice}')
#         value = row['answer']
        
#         # Check if the extracted answer is correct
#         if choice == str(chr(65 + value)):
#             return 1  # Return 1 if correct
#     else:
#         print("No match found")
    
#     return 0  # Return 0 if no match or incorrect answer

# def run_in_parallel(filtered_test_df):
#     count = 0
    
#     # Use ThreadPoolExecutor for parallel execution
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Create a list of futures
#         futures = [executor.submit(process_row, index, row) for index, row in filtered_test_df.iterrows()]
        
#         # Gather the results as they complete
#         for future in concurrent.futures.as_completed(futures):
#             count += future.result()  # Add the result (1 or 0) to count
    
#     return count

# # Run the parallel processing
# correct_count = run_in_parallel(filtered_test_df)
# print(f'Total correct answers: {correct_count}')


# count =0
# for index, row in filtered_test_df.iterrows():
#         question = row['question']
#         options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
#         torch.cuda.empty_cache()
#         gc.collect()
#         answer, _, _ = medrag.answer(question=question, options=options, k=12)
#         choice = locate_answer(answer)
#         value = row['answer']
#         if choice==str(chr(65 + value)):
#                  count+=1
#         print(f'Extracted answer: {choice} and actual answer: {str(chr(65 + value))}')
#         # torch.cuda.empty_cache()
#         # gc.collect()

#         # print(answer)
#         # pattern = r'(?i)(answer[_ ]?choice|best answer is|correct answer is)\W?["\']?\s*([A-D])["\']?'

#        # Extract answer choices
#         # match = re.search(pattern, answer)
#         # if match:
#         #     choice = {match.group(2)}
#         #    # print(f'Extracted answer: {choice}')
#         #     value = row['answer']
#         #     if choice==str(chr(65 + value)):
#         #         count+=1
#         #     print(f'Extracted answer: {choice} and actual answer: {str(chr(65 + value))}')
#         # else:
#         #     print("No match found")
        
#         time.sleep(1)

# print(count)
# answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
# print(f"Final answer in json with rationale: {answer}")
# {
#   "step_by_step_thinking": "A lesion causing compression of the facial nerve at the stylomastoid foramen will result in paralysis of the facial muscles. Loss of taste, lacrimation, and decreased salivation are not specifically mentioned in relation to a lesion at the stylomastoid foramen.", 
#   "answer_choice": "A"
# }

# ### MedRAG with pre-determined snippets
# snippets = [{'id': 'InternalMed_Harrison_30037', 'title': 'InternalMed_Harrison', 'content': 'On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.', 'contents': 'InternalMed_Harrison. On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.'}]
# answer, _, _ = medrag.answer(question=question, options=options, snippets=snippets)

# ### MedRAG with pre-determined snippet ids
# snippets_ids = [{"id": s["id"]} for s in snippets]
# answer, snippets, _ = medrag.answer(question=question, options=options, snippets_ids=snippets_ids)


# # Initialize variables
# count = 0
# questions_list = []
# options_list = []
# correct_answers_list = []

# # Collect questions, options, and correct answers into lists
# for index, row in filtered_test_df.iterrows():
#     question = row['question']
#     options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
#     value = row['answer']  # Assuming 'answer' is an integer index starting from 0

#     questions_list.append(question)
#     options_list.append(options)
#     correct_answers_list.append(chr(65 + value))  # Convert index to corresponding letter

# # Define batch size to manage memory usage
# batch_size = 10  # You can adjust this based on your GPU memory

# # Process questions in batches
# for batch_start in range(0, len(questions_list), batch_size):
#     batch_end = batch_start + batch_size
#     batch_questions = questions_list[batch_start:batch_end]
#     batch_options = options_list[batch_start:batch_end]
#     batch_correct_answers = correct_answers_list[batch_start:batch_end]

#     torch.cuda.empty_cache()
#     gc.collect()

#     # Get answers using the updated method
#     answers, snippets_list, scores_list = medrag.answer(
#         questions=batch_questions,
#         options_list=batch_options,
#         k=16
#     )

#     # Iterate over the batch to compare predicted and actual answers
#     for i, answer in enumerate(answers):
#         choice = locate_answer(answer)
#         correct_choice = batch_correct_answers[i]
#         if choice == correct_choice:
#             count += 1
#         print(f'Extracted answer: {choice} and actual answer: {correct_choice}')

#     torch.cuda.empty_cache()
#     gc.collect()

# print(f'Total correct answers: {count}')




# count = 0
# questions_list = []
# options_list = []
# correct_answers_list = []

#     # Function to process a single row
# def process_question(args):
#         index, row = args
#         question = row['question']
#         options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
#         value = row['answer']  # Assuming 'answer' is an integer index starting from 0
#         correct_answer = chr(65 + value)
#         return question, options, correct_answer

#     # Prepare the arguments
# args_list = list(filtered_test_df.iterrows())

#     # Use multiprocessing to process questions
# with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(process_question, args_list)

#     # Unpack the results
# questions_list, options_list, correct_answers_list = zip(*results)

#     # Create dataset and dataloader
# class QuestionDataset(Dataset):
#         def __init__(self, questions_list, options_list, correct_answers_list):
#             self.questions = questions_list
#             self.options = options_list
#             self.correct_answers = correct_answers_list

#         def __len__(self):
#             return len(self.questions)

#         def __getitem__(self, idx):
#             return {
#                 'question': self.questions[idx],
#                 'options': self.options[idx],
#                 'correct_answer': self.correct_answers[idx]
#             }

# dataset = QuestionDataset(questions_list, options_list, correct_answers_list)
# batch_size = 32  # Adjust based on GPU memory
# dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

# count = 0

#     # Process batches
# for batch in dataloader:
#         batch_questions = batch['question']
#         batch_options = batch['options']
#         batch_correct_answers = batch['correct_answer']

#         torch.cuda.empty_cache()
#         gc.collect()

#         # Get answers using the updated method
#         answers, snippets_list, scores_list = medrag.answer(
#             questions=batch_questions,
#             options_list=batch_options,
#             k=16
#         )

#         # Iterate over the batch to compare predicted and actual answers
#         for i, answer in enumerate(answers):
#             choice = locate_answer(answer)
#             correct_choice = batch_correct_answers[i]
#             if choice == correct_choice:
#                 count += 1
#             print(f'Extracted answer: {choice} and actual answer: {correct_choice}')

#         torch.cuda.empty_cache()
#         gc.collect()

# print(f'Total correct answers: {count}')


from multiprocessing import Process,set_start_method

medrag = MedRAG(llm_name="meta-llama/Llama-3.2-1B-Instruct", rag=True, retriever_name="Contriever", corpus_name="StatPearls")

desired_subjects = ['anatomy', 'clinical_knowledge', 'professional_medicine', 'human_genetics', 'college_medicine', 'college_biology']
dataset = load_dataset('cais/mmlu', "all")

def filter_and_convert_to_pandas(ds, desired_subjects):
        # Filter the dataset
        filtered_ds = ds.filter(lambda row: row['subject'] in desired_subjects)
        # Convert the filtered dataset to Pandas DataFrame
        return filtered_ds.to_pandas()

filtered_datasets = {}


def process_batch(batch_df):
    count = 0
    for index, row in batch_df.iterrows():
        question = row['question']  # Adjust column name as necessary
        options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
        torch.cuda.empty_cache()
        gc.collect()
        answer, _, _ = medrag.answer(question=question, options=options, k=8)
        choice = locate_answer(answer)
        value = row['answer']
        if choice == str(str(chr(65 + value))):
            count += 1
        print(f'Extracted answer: {choice} and actual answer: {str(chr(65 + value))}')
        time.sleep(1)
    print(f'Batch processed. Correct answers: {count}')

if __name__ == "__main__":

    try:
            set_start_method('spawn', force=True)  # Ensure 'spawn' is used
    except RuntimeError:
            pass
        # Split data into batches

    for split in dataset:
        print(f"Processing {split} split...")
        # Filter and convert each split to a DataFrame
        filtered_datasets[split] = filter_and_convert_to_pandas(dataset[split], desired_subjects)

# Now `filtered_datasets` is a dictionary where each key (split) has a filtered Pandas DataFrame
# For example, to get the DataFrame for the "test" split:
    filtered_test_df = filtered_datasets['test']

    batch_size = 150
    batches = [filtered_test_df.iloc[i:i+batch_size] for i in range(0, len(filtered_test_df), batch_size)]

    # Process each batch in separate processes
    for batch_df in batches:
        p = Process(target=process_batch, args=(batch_df,))
        #,))
    # ,))
        p.start()
        p.join()  # Wait for the process to finish before starting the next one
    #``

#print("All batches have been processed.")

# def process_batch(batch_df, batch_num, return_dict):
#     count = 0
#     for index, row in batch_df.iterrows():
#         question = row['question']
#         options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
#         torch.cuda.empty_cache()
#         gc.collect()
#         answer, _, _ = medrag.answer(question=question, options=options, k=12)
#         choice = locate_answer(answer)
#         value = row['answer']
#         if choice == str(chr(65 + value)):
#             count += 1
#         print(f'Extracted answer: {choice} and actual answer: {str(chr(65 + value))}')
#         time.sleep(1)
#     print(f'Count in batch {batch_num}: {count}')
#     # Store the count in the shared dictionary
#     return_dict[batch_num] = count




# try:
#         set_start_method('spawn', force=True)  # Ensure 'spawn' is used
# except RuntimeError:
#         pass
# batch_size = 150
# total_rows = len(filtered_test_df)
# num_batches = total_rows // batch_size + (1 if total_rows % batch_size > 0 else 0)
# manager = Manager()
# return_dict = manager.dict()

# for i in range(num_batches):
#         # Slice the DataFrame for the current batch
#         batch_df = filtered_test_df.iloc[i*batch_size:(i+1)*batch_size]
#         p = Process(target=process_batch, args=(batch_df, i, return_dict))
#         p.start()
#         p.join()  # Wait for the process to finish before starting the next one

#     # Sum the counts from all batches
# total_count = sum(return_dict.values())
# print(f'Total correct answers: {total_count}')

