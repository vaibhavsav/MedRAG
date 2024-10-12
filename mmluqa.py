from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

cot = MedRAG(llm_name="meta-llama/Llama-3.2-1B-Instruct", rag=False)
answer, _, _ = cot.answer(question=question, options=options)
print(f"Final answer in json with rationale: {answer}")
