from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

medrag = MedRAG(llm_name="meta-llama/Llama-3.2-1B-Instruct", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
print(f"Final answer in json with rationale: {answer}")
# {
#   "step_by_step_thinking": "A lesion causing compression of the facial nerve at the stylomastoid foramen will result in paralysis of the facial muscles. Loss of taste, lacrimation, and decreased salivation are not specifically mentioned in relation to a lesion at the stylomastoid foramen.", 
#   "answer_choice": "A"
# }

### MedRAG with pre-determined snippets
snippets = [{'id': 'InternalMed_Harrison_30037', 'title': 'InternalMed_Harrison', 'content': 'On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.', 'contents': 'InternalMed_Harrison. On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.'}]
answer, _, _ = medrag.answer(question=question, options=options, snippets=snippets)

### MedRAG with pre-determined snippet ids
snippets_ids = [{"id": s["id"]} for s in snippets]
answer, snippets, _ = medrag.answer(question=question, options=options, snippets_ids=snippets_ids)
