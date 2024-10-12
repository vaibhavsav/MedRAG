from src.medrag import MedRAG
from datasets import load_dataset

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

medrag = MedRAG(llm_name="meta-llama/Llama-3.2-1B-Instruct", rag=True, retriever_name="Contriever", corpus_name="StatPearls")

desired_subjects = ['anatomy', 'clinical_knowledge', 'professional_medicine', 'human_genetics', 'college_medicine', 'college_biology']
dataset = load_dataset('cais/mmlu', "all")

def filter_and_convert_to_pandas(ds, desired_subjects):
    # Filter the dataset
    filtered_ds = ds.filter(lambda row: row['subject'] in desired_subjects)
    # Convert the filtered dataset to Pandas DataFrame
    return filtered_ds.to_pandas()

filtered_datasets = {}

for split in dataset:
    print(f"Processing {split} split...")
    # Filter and convert each split to a DataFrame
    filtered_datasets[split] = filter_and_convert_to_pandas(dataset[split], desired_subjects)

# Now `filtered_datasets` is a dictionary where each key (split) has a filtered Pandas DataFrame
# For example, to get the DataFrame for the "test" split:
filtered_test_df = filtered_datasets['test']

for index, row in filtered_test_df.iterrows():
        question = row['question']
        options = {chr(65 + i): option for i, option in enumerate(row['choices'])}
        answer, snippets, scores = medrag.answer(question=question, options=options, k=32)
        print(answer)


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
