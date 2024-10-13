import os
import re
import json
import torch
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *
from config import config

# OpenAI API configuration
openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in self.llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in self.llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in self.llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in self.llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in self.llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
                
            )
        
        self.follow_up = follow_up
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        else:
            self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages_list):
        '''
        Generate responses given a list of messages.
        messages_list: List of messages, each is a list of dicts with 'role' and 'content'.
        '''
        answers = []
        if "openai" in self.llm_name.lower():
            for messages in messages_list:
                ans = openai_client(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                answers.append(ans)
        elif "gemini" in self.llm_name.lower():
            for messages in messages_list:
                response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"])
                ans = response.candidates[0].content.parts[0].text
                answers.append(ans)
        else:
            for messages in messages_list:
                stopping_criteria = None
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if "meditron" in self.llm_name.lower():
                    stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
                if "llama-3" in self.llm_name.lower():
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=self.max_length,
                        truncation=True,
                        stopping_criteria=stopping_criteria
                    )
                else:
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=self.max_length,
                        truncation=True,
                        stopping_criteria=stopping_criteria
                    )
                ans = response[0]["generated_text"][len(prompt):]
                answers.append(ans)
        return answers

    def medrag_answer(self, questions, options_list=None, k=32, rrf_k=100, save_dir=None, snippets_list=None, snippets_ids_list=None, **kwargs):
        '''
        questions (List[str]): List of questions to be answered.
        options_list (List[Dict[str, str]]): List of options for each question.
        k (int): Number of snippets to retrieve.
        rrf_k (int): Parameter for Reciprocal Rank Fusion.
        save_dir (str): Directory to save the results.
        snippets_list (List[List[Dict]]): List of snippets for each question.
        snippets_ids_list (List[List[Dict]]): List of snippet IDs for each question.
        '''

        # Ensure options_list, snippets_list, and snippets_ids_list are lists with the same length as questions
        num_questions = len(questions)
        options_list = options_list if options_list is not None else [None] * num_questions
        snippets_list = snippets_list if snippets_list is not None else [None] * num_questions
        snippets_ids_list = snippets_ids_list if snippets_ids_list is not None else [None] * num_questions

        all_answers = []
        all_retrieved_snippets = []
        all_scores = []

        for idx, question in enumerate(questions):
            options = options_list[idx]
            snippets = snippets_list[idx]
            snippets_ids = snippets_ids_list[idx]

            if options is not None:
                options_str = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
            else:
                options_str = ''

            # Retrieve relevant snippets
            if self.rag:
                if snippets is not None:
                    retrieved_snippets = snippets[:k]
                    scores = []
                elif snippets_ids is not None:
                    if self.docExt is None:
                        self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                    retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                    scores = []
                else:
                    assert self.retrieval_system is not None
                    retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

                contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
                if len(contexts) == 0:
                    contexts = [""]
                if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
                    contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
                else:
                    contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
            else:
                retrieved_snippets = []
                scores = []
                contexts = [""]

            if save_dir is not None and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Generate answers
            messages_list = []
            if not self.rag:
                prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_str)
                messages = [
                    {"role": "system", "content": self.templates["cot_system"]},
                    {"role": "user", "content": prompt_cot}
                ]
                messages_list.append(messages)
            else:
                for context in contexts:
                    prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options_str)
                    messages = [
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                    ]
                    messages_list.append(messages)

            ans_list = self.generate(messages_list)
            ans_list = [re.sub(r"\s+", " ", ans) for ans in ans_list]

            if save_dir is not None:
                question_save_dir = os.path.join(save_dir, f"question_{idx}")
                if not os.path.exists(question_save_dir):
                    os.makedirs(question_save_dir)
                with open(os.path.join(question_save_dir, "snippets.json"), 'w') as f:
                    json.dump(retrieved_snippets, f, indent=4)
                with open(os.path.join(question_save_dir, "response.json"), 'w') as f:
                    json.dump(ans_list, f, indent=4)

            all_answers.append(ans_list[0] if len(ans_list) == 1 else ans_list)
            all_retrieved_snippets.append(retrieved_snippets)
            all_scores.append(scores)

        return all_answers, all_retrieved_snippets, all_scores

    def i_medrag_answer(self, questions, options_list=None, k=32, rrf_k=100, save_paths=None, n_rounds=4, n_queries=3, qa_cache_paths=None, **kwargs):
        '''
        questions (List[str]): List of questions to be answered.
        options_list (List[Dict[str, str]]): List of options for each question.
        k (int): Number of snippets to retrieve.
        rrf_k (int): Parameter for Reciprocal Rank Fusion.
        save_paths (List[str]): List of file paths to save the results for each question.
        n_rounds (int): Number of interactive rounds.
        n_queries (int): Number of queries per round.
        qa_cache_paths (List[str]): List of cache file paths for each question.
        '''

        num_questions = len(questions)
        options_list = options_list if options_list is not None else [None] * num_questions
        save_paths = save_paths if save_paths is not None else [None] * num_questions
        qa_cache_paths = qa_cache_paths if qa_cache_paths is not None else [None] * num_questions

        all_answers = []
        all_messages = []

        for idx, question in enumerate(questions):
            options = options_list[idx]
            save_path = save_paths[idx] if save_paths else None
            qa_cache_path = qa_cache_paths[idx] if qa_cache_paths else None

            if options is not None:
                options_str = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
            else:
                options_str = ''
            QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options_str}"

            context = ""
            qa_cache = []
            if qa_cache_path is not None and os.path.exists(qa_cache_path):
                qa_cache = eval(open(qa_cache_path, 'r').read())[:n_rounds]
                if len(qa_cache) > 0:
                    context = qa_cache[-1]
                n_rounds_remaining = n_rounds - len(qa_cache)
            else:
                n_rounds_remaining = n_rounds
            last_context = None

            # Run in loop
            max_iterations = n_rounds_remaining + 3
            saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

            for i in range(max_iterations):
                if i < n_rounds_remaining:
                    if context == "":
                        messages = [
                            {
                                "role": "system",
                                "content": self.templates["i_medrag_system"],
                            },
                            {
                                "role": "user",
                                "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                            },
                        ]
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": self.templates["i_medrag_system"],
                            },
                            {
                                "role": "user",
                                "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                            },
                        ]
                elif context != last_context:
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                        },
                    ]
                elif len(messages) == 1:
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                        },
                    ]
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                last_context = context
                last_content = self.generate([messages])[0]
                response_message = {"role": "assistant", "content": last_content}
                saved_messages.append(response_message)
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                if i >= n_rounds_remaining and ("## Answer" in last_content or "answer is" in last_content.lower()):
                    messages.append(response_message)
                    messages.append(
                        {
                            "role": "user",
                            "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options_str else "Output the answer in JSON: {'answer': your_answer}",
                        }
                    )
                    saved_messages.append(messages[-1])
                    answer_content = self.generate([messages])[0]
                    answer_message = {"role": "assistant", "content": answer_content}
                    messages.append(answer_message)
                    saved_messages.append(messages[-1])
                    if save_path:
                        with open(save_path, 'w') as f:
                            json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                    all_answers.append(messages[-1]["content"])
                    all_messages.append(messages)
                    break
                elif "## Queries" in last_content:
                    messages = messages[:-1]
                    if last_content.split("## Queries")[-1].strip() == "":
                        print("Empty queries. Continue with next iteration.")
                        continue
                    try:
                        action_str = self.generate([
                            {
                                "role": "user",
                                "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                            },
                        ])[0]
                        action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                        action_list = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in eval(action_str)]
                    except Exception as E:
                        print("Error parsing action list. Continue with next iteration.")
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                        continue
                    for sub_question in action_list:
                        if sub_question.strip() == "":
                            continue
                        try:
                            rag_result = self.medrag_answer([sub_question], k=k, rrf_k=rrf_k)[0][0]
                            context += f"\n\nQuery: {sub_question}\nAnswer: {rag_result}"
                            context = context.strip()
                        except Exception as E:
                            error_class = E.__class__.__name__
                            error = f"{error_class}: {str(E)}"
                            print(error)
                            if save_path:
                                with open(save_path + ".error", 'a') as f:
                                    f.write(f"{error}\n")
                    qa_cache.append(context)
                    if qa_cache_path:
                        with open(qa_cache_path, 'w') as f:
                            json.dump(qa_cache, f, indent=4)
                else:
                    messages.append(response_message)
                    print("No queries or answer. Continue with next iteration.")
                    continue

        return all_answers, all_messages

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
