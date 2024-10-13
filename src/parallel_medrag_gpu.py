# Import necessary libraries
import os
import re
import json
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
import gc
import multiprocessing as mp


# Assuming 'src' directory contains required modules
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *
from config import config

# Configure OpenAI API
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

# Custom Stopping Criteria
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)

# Define MedRAG class with optimizations
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
        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag,
        }
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

            # Load model for multiple GPUs using DataParallel
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
                # Replace with your actual Hugging Face token
                use_auth_token="YOUR_HF_TOKEN"
            )

            # Wrap the model with DataParallel
            model = torch.nn.DataParallel(model)
            model = model.to('cuda')

            self.model = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device='cuda:0',
                framework='pt',
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
        if "openai" in self.llm_name.lower():
            answers = []
            for messages in messages_list:
                ans = openai_client(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                answers.append(ans)
        elif "gemini" in self.llm_name.lower():
            answers = []
            for messages in messages_list:
                response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"])
                ans = response.candidates[0].content.parts[0].text
                answers.append(ans)
        else:
            # Prepare inputs for the model
            inputs = []
            for messages in messages_list:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs.append(prompt)

            # Tokenize the inputs
            encoded_inputs = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.context_length,
                return_tensors="pt"
            ).to('cuda')

            # Generate outputs in parallel
            with torch.no_grad():
                outputs = self.model.model.generate(
                    input_ids=encoded_inputs['input_ids'],
                    attention_mask=encoded_inputs['attention_mask'],
                    max_length=self.max_length,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode outputs
            answers = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the prompt from the generated text
                answer = generated_text[len(inputs[i]):]
                answers.append(answer)
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
        messages_list = []

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

                contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(i, retrieved_snippets[i]["title"], retrieved_snippets[i]["content"]) for i in range(len(retrieved_snippets))]
                if len(contexts) == 0:
                    contexts = [""]
                if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
                    context_str = self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])
                else:
                    context_str = self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])
            else:
                retrieved_snippets = []
                scores = []
                context_str = ""

            if save_dir is not None and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Generate messages
            if not self.rag:
                prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_str)
                messages = [
                    {"role": "system", "content": self.templates["cot_system"]},
                    {"role": "user", "content": prompt_cot}
                ]
                messages_list.append(messages)
            else:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context_str, question=question, options=options_str)
                messages = [
                    {"role": "system", "content": self.templates["medrag_system"]},
                    {"role": "user", "content": prompt_medrag}
                ]
                messages_list.append(messages)

            all_retrieved_snippets.append(retrieved_snippets)
            all_scores.append(scores)

        # Generate answers
        ans_list = self.generate(messages_list)
        ans_list = [re.sub(r"\s+", " ", ans) for ans in ans_list]

        # Save results
        if save_dir is not None:
            for idx, ans in enumerate(ans_list):
                question_save_dir = os.path.join(save_dir, f"question_{idx}")
                if not os.path.exists(question_save_dir):
                    os.makedirs(question_save_dir)
                with open(os.path.join(question_save_dir, "snippets.json"), 'w') as f:
                    json.dump(all_retrieved_snippets[idx], f, indent=4)
                with open(os.path.join(question_save_dir, "response.json"), 'w') as f:
                    json.dump(ans, f, indent=4)

        return ans_list, all_retrieved_snippets, all_scores
