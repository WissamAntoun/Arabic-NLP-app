import json
import os
from typing import List
import logging
import more_itertools
import pandas as pd
import requests
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed

from .modeling_gpt2 import GPT2LMHeadModel as GROVERLMHeadModel
from .preprocess import ArabertPreprocessor
from .sa_utils import *
from .utils import download_models, softmax

from functools import lru_cache
from urllib.parse import unquote

import streamlit as st
import wikipedia
from codetiming import Timer
from fuzzysearch import find_near_matches
from googleapi import google
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
# Taken and Modified from https://huggingface.co/spaces/flax-community/chef-transformer/blob/main/app.py
class TextGeneration:
    def __init__(self):
        self.debug = False
        self.generation_pipline = {}
        self.preprocessor = ArabertPreprocessor(model_name="aragpt2-mega")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "aubmindlab/aragpt2-mega", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.API_KEY = os.getenv("API_KEY")
        self.headers = {"Authorization": f"Bearer {self.API_KEY}"}
        # self.model_names_or_paths = {
        #     "aragpt2-medium": "D:/ML/Models/aragpt2-medium",
        #     "aragpt2-base": "D:/ML/Models/aragpt2-base",
        # }
        self.model_names_or_paths = {
            "aragpt2-medium": "aubmindlab/aragpt2-medium",
            "aragpt2-base": "aubmindlab/aragpt2-base",
            "aragpt2-large": "aubmindlab/aragpt2-large",
            "aragpt2-mega": "aubmindlab/aragpt2-mega",
        }
        set_seed(42)

    def load_pipeline(self):
        for model_name, model_path in self.model_names_or_paths.items():
            if "base" in model_name or "medium" in model_name:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GPT2LMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )
            else:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GROVERLMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )

    def load(self):
        if not self.debug:
            self.load_pipeline()

    def generate(
        self,
        model_name,
        prompt,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        do_sample: bool,
        num_beams: int,
    ):
        logger.info(f"Generating with {model_name}")
        prompt = self.preprocessor.preprocess(prompt)
        return_full_text = False
        return_text = True
        num_return_sequences = 1
        pad_token_id = 0
        eos_token_id = 0
        input_tok = self.tokenizer.tokenize(prompt)
        max_length = len(input_tok) + max_new_tokens
        if max_length > 1024:
            max_length = 1024
        if not self.debug:
            generated_text = self.generation_pipline[model_name.lower()](
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )[0]["generated_text"]
        else:
            generated_text = self.generate_by_query(
                prompt,
                model_name,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )
            # print(generated_text)
            if isinstance(generated_text, dict):
                if "error" in generated_text:
                    if "is currently loading" in generated_text["error"]:
                        return f"Model is currently loading, estimated time is {generated_text['estimated_time']}"
                    return generated_text["error"]
                else:
                    return "Something happened 🤷‍♂️!!"
            else:
                generated_text = generated_text[0]["generated_text"]

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")
        return self.preprocessor.unpreprocess(generated_text)

    def query(self, payload, model_name):
        data = json.dumps(payload)
        url = (
            "https://api-inference.huggingface.co/models/aubmindlab/"
            + model_name.lower()
        )
        response = requests.request("POST", url, headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def generate_by_query(
        self,
        prompt: str,
        model_name: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        pad_token_id: int,
        eos_token_id: int,
        return_full_text: int,
        return_text: int,
        do_sample: bool,
        num_beams: int,
        num_return_sequences: int,
    ):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length ": max_length,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "return_full_text": return_full_text,
                "return_text": return_text,
                "pad_token_id": pad_token_id,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
            },
            "options": {
                "use_cache": True,
            },
        }
        return self.query(payload, model_name)


class SentimentAnalyzer:
    def __init__(self):
        self.sa_models = [
            "sa_trial5_1",
            "sa_no_aoa_in_neutral",
            "sa_cnnbert",
            "sa_sarcasm",
            "sar_trial10",
            "sa_no_AOA",
        ]
        download_models(self.sa_models)
        # fmt: off
        self.processors = {
            "sa_trial5_1": Trial5ArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            "sa_no_aoa_in_neutral": NewArabicPreprocessorBalanced(model_name='UBC-NLP/MARBERT'),
            "sa_cnnbert": CNNMarbertArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            "sa_sarcasm": SarcasmArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            "sar_trial10": SarcasmArabicPreprocessor(model_name='UBC-NLP/MARBERT'),
            "sa_no_AOA": NewArabicPreprocessorBalanced(model_name='UBC-NLP/MARBERT'),
        }

        self.pipelines = {
            "sa_trial5_1": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_trial5_1",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_trial5_1")],
            "sa_no_aoa_in_neutral": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_no_aoa_in_neutral",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_no_aoa_in_neutral")],
            "sa_cnnbert": [CNNTextClassificationPipeline("{}/train_{}/best_model".format("sa_cnnbert",i), device=-1, return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_cnnbert")],
            "sa_sarcasm": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_sarcasm",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_sarcasm")],
            "sar_trial10": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sar_trial10",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sar_trial10")],
            "sa_no_AOA": [pipeline("sentiment-analysis", model="{}/train_{}/best_model".format("sa_no_AOA",i), device=-1,return_all_scores =True) for i in tqdm(range(0,5), desc=f"Loading pipeline for model: sa_no_AOA")],
        }
        # fmt: on

    def get_preds_from_sarcasm(self, texts):
        prep = self.processors["sar_trial10"]
        prep_texts = [prep.preprocess(x) for x in texts]

        preds_df = pd.DataFrame([])
        for i in range(0, 5):
            preds = []
            for s in more_itertools.chunked(list(prep_texts), 128):
                preds.extend(self.pipelines["sar_trial10"][i](s))
            preds_df[f"model_{i}"] = preds

        final_labels = []
        final_scores = []
        for id, row in preds_df.iterrows():
            pos_total = 0
            neu_total = 0
            for pred in row[:]:
                pos_total += pred[0]["score"]
                neu_total += pred[1]["score"]

            pos_avg = pos_total / len(row[:])
            neu_avg = neu_total / len(row[:])

            final_labels.append(
                self.pipelines["sar_trial10"][0].model.config.id2label[
                    np.argmax([pos_avg, neu_avg])
                ]
            )
            final_scores.append(np.max([pos_avg, neu_avg]))

        return final_labels, final_scores

    def get_preds_from_a_model(self, texts: List[str], model_name):
        try:
            prep = self.processors[model_name]

            prep_texts = [prep.preprocess(x) for x in texts]
            if model_name == "sa_sarcasm":
                sarcasm_label, _ = self.get_preds_from_sarcasm(texts)
                sarcastic_map = {"Not_Sarcastic": "غير ساخر", "Sarcastic": "ساخر"}
                labeled_prep_texts = []
                for t, l in zip(prep_texts, sarcasm_label):
                    labeled_prep_texts.append(sarcastic_map[l] + " [SEP] " + t)

            preds_df = pd.DataFrame([])
            for i in range(0, 5):
                preds = []
                for s in more_itertools.chunked(list(prep_texts), 128):
                    preds.extend(self.pipelines[model_name][i](s))
                preds_df[f"model_{i}"] = preds

            final_labels = []
            final_scores = []
            final_scores_list = []
            for id, row in preds_df.iterrows():
                pos_total = 0
                neg_total = 0
                neu_total = 0
                for pred in row[2:]:
                    pos_total += pred[0]["score"]
                    neu_total += pred[1]["score"]
                    neg_total += pred[2]["score"]

                pos_avg = pos_total / 5
                neu_avg = neu_total / 5
                neg_avg = neg_total / 5

                if model_name == "sa_no_aoa_in_neutral":
                    final_labels.append(
                        self.pipelines[model_name][0].model.config.id2label[
                            np.argmax([neu_avg, neg_avg, pos_avg])
                        ]
                    )
                else:
                    final_labels.append(
                        self.pipelines[model_name][0].model.config.id2label[
                            np.argmax([pos_avg, neu_avg, neg_avg])
                        ]
                    )
                final_scores.append(np.max([pos_avg, neu_avg, neg_avg]))
                final_scores_list.append((pos_avg, neu_avg, neg_avg))
        except RuntimeError as e:
            if model_name == "sa_cnnbert":
                return (
                    ["Neutral"] * len(texts),
                    [0.0] * len(texts),
                    [(0.0, 0.0, 0.0)] * len(texts),
                )
            else:
                raise RuntimeError(e)
        return final_labels, final_scores, final_scores_list

    def predict(self, texts: List[str]):
        logger.info(f"Predicting for: {texts}")
        (
            new_balanced_label,
            new_balanced_score,
            new_balanced_score_list,
        ) = self.get_preds_from_a_model(texts, "sa_no_aoa_in_neutral")
        (
            cnn_marbert_label,
            cnn_marbert_score,
            cnn_marbert_score_list,
        ) = self.get_preds_from_a_model(texts, "sa_cnnbert")
        trial5_label, trial5_score, trial5_score_list = self.get_preds_from_a_model(
            texts, "sa_trial5_1"
        )
        no_aoa_label, no_aoa_score, no_aoa_score_list = self.get_preds_from_a_model(
            texts, "sa_no_AOA"
        )
        sarcasm_label, sarcasm_score, sarcasm_score_list = self.get_preds_from_a_model(
            texts, "sa_sarcasm"
        )

        id_label_map = {0: "Positive", 1: "Neutral", 2: "Negative"}

        final_ensemble_prediction = []
        final_ensemble_score = []
        final_ensemble_all_score = []
        for entry in zip(
            new_balanced_score_list,
            cnn_marbert_score_list,
            trial5_score_list,
            no_aoa_score_list,
            sarcasm_score_list,
        ):
            pos_score = 0
            neu_score = 0
            neg_score = 0
            for s in entry:
                pos_score += s[0] * 1.57
                neu_score += s[1] * 0.98
                neg_score += s[2] * 0.93

                # weighted 2
                # pos_score += s[0]*1.67
                # neu_score += s[1]
                # neg_score += s[2]*0.95

            final_ensemble_prediction.append(
                id_label_map[np.argmax([pos_score, neu_score, neg_score])]
            )
            final_ensemble_score.append(np.max([pos_score, neu_score, neg_score]))
            final_ensemble_all_score.append(
                softmax(np.array([pos_score, neu_score, neg_score])).tolist()
            )

        logger.info(f"Result: {final_ensemble_prediction}")
        logger.info(f"Score: {final_ensemble_score}")
        logger.info(f"All Scores: {final_ensemble_all_score}")
        return final_ensemble_prediction, final_ensemble_score, final_ensemble_all_score


wikipedia.set_lang("ar")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

preprocessor = ArabertPreprocessor("wissamantoun/araelectra-base-artydiqa")
logger.info("Loading QA Pipeline...")
tokenizer = AutoTokenizer.from_pretrained("wissamantoun/araelectra-base-artydiqa")
qa_pipe = pipeline("question-answering", model="wissamantoun/araelectra-base-artydiqa")
logger.info("Finished loading QA Pipeline...")


@lru_cache(maxsize=100)
def get_qa_answers(question):
    logger.info("\n=================================================================")
    logger.info(f"Question: {question}")

    if "وسام أنطون" in question or "wissam antoun" in question.lower():
        return {
            "title": "Creator",
            "results": [
                {
                    "score": 1.0,
                    "new_start": 0,
                    "new_end": 12,
                    "new_answer": "My Creator 😜",
                    "original": "My Creator 😜",
                    "link": "https://github.com/WissamAntoun/",
                }
            ],
        }
    search_timer = Timer(
        "search and wiki", text="Search and Wikipedia Time: {:.2f}", logger=logging.info
    )
    try:
        search_timer.start()
        search_results = google.search(
            question + " site:ar.wikipedia.org", lang="ar", area="ar"
        )
        if len(search_results) == 0:
            return {}

        page_name = search_results[0].link.split("wiki/")[-1]
        wiki_page = wikipedia.page(unquote(page_name))
        wiki_page_content = wiki_page.content
        search_timer.stop()
    except:
        return {}

    sections = []
    for section in re.split("== .+ ==[^=]", wiki_page_content):
        if not section.isspace():
            prep_section = tokenizer.tokenize(preprocessor.preprocess(section))
            if len(prep_section) > 500:
                subsections = []
                for subsection in re.split("=== .+ ===", section):
                    if subsection.isspace():
                        continue
                    prep_subsection = tokenizer.tokenize(
                        preprocessor.preprocess(subsection)
                    )
                    subsections.append(subsection)
                    # logger.info(f"Subsection found with length: {len(prep_subsection)}")
                sections.extend(subsections)
            else:
                # logger.info(f"Regular Section with length: {len(prep_section)}")
                sections.append(section)

    full_len_sections = []
    temp_section = ""
    for section in sections:
        if (
            len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))
            + len(tokenizer.tokenize(preprocessor.preprocess(section)))
            > 384
        ):
            if temp_section == "":
                temp_section = section
                continue
            full_len_sections.append(temp_section)
            # logger.info(
            #     f"full section length: {len(tokenizer.tokenize(preprocessor.preprocess(temp_section)))}"
            # )
            temp_section = ""
        else:
            temp_section += " " + section + " "
    if temp_section != "":
        full_len_sections.append(temp_section)

    reader_time = Timer("electra", text="Reader Time: {:.2f}", logger=logging.info)
    reader_time.start()
    results = qa_pipe(
        question=[preprocessor.preprocess(question)] * len(full_len_sections),
        context=[preprocessor.preprocess(x) for x in full_len_sections],
    )

    if not isinstance(results, list):
        results = [results]

    logger.info(f"Wiki Title: {unquote(page_name)}")
    logger.info(f"Total Sections: {len(sections)}")
    logger.info(f"Total Full Sections: {len(full_len_sections)}")

    for result, section in zip(results, full_len_sections):
        result["original"] = section
        answer_match = find_near_matches(
            " " + preprocessor.unpreprocess(result["answer"]) + " ",
            result["original"],
            max_l_dist=min(5, len(preprocessor.unpreprocess(result["answer"])) // 2),
            max_deletions=0,
        )
        try:
            result["new_start"] = answer_match[0].start
            result["new_end"] = answer_match[0].end
            result["new_answer"] = answer_match[0].matched
            result["link"] = (
                search_results[0].link + "#:~:text=" + result["new_answer"].strip()
            )
        except:
            result["new_start"] = result["start"]
            result["new_end"] = result["end"]
            result["new_answer"] = result["answer"]
            result["original"] = preprocessor.preprocess(result["original"])
            result["link"] = search_results[0].link
        logger.info(f"Answers: {preprocessor.preprocess(result['new_answer'])}")

    sorted_results = sorted(results, reverse=True, key=lambda x: x["score"])

    return_dict = {}
    return_dict["title"] = unquote(page_name)
    return_dict["results"] = sorted_results

    reader_time.stop()
    logger.info(f"Total time spent: {reader_time.last + search_timer.last}")
    return return_dict
