import re
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fuzzysearch import find_near_matches
from pyarabic import araby
from torch import nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, pipeline
from transformers.modeling_outputs import SequenceClassifierOutput

from .preprocess import ArabertPreprocessor, url_regexes, user_mention_regex

multiple_char_pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

# ASAD-NEW_AraBERT_PREP-Balanced
class NewArabicPreprocessorBalanced(ArabertPreprocessor):
    def __init__(
        self,
        model_name: str,
        keep_emojis: bool = False,
        remove_html_markup: bool = True,
        replace_urls_emails_mentions: bool = True,
        strip_tashkeel: bool = True,
        strip_tatweel: bool = True,
        insert_white_spaces: bool = True,
        remove_non_digit_repetition: bool = True,
        replace_slash_with_dash: bool = None,
        map_hindi_numbers_to_arabic: bool = None,
        apply_farasa_segmentation: bool = None,
    ):
        if "UBC-NLP" in model_name or "CAMeL-Lab" in model_name:
            keep_emojis = True
            remove_non_digit_repetition = True
        super().__init__(
            model_name=model_name,
            keep_emojis=keep_emojis,
            remove_html_markup=remove_html_markup,
            replace_urls_emails_mentions=replace_urls_emails_mentions,
            strip_tashkeel=strip_tashkeel,
            strip_tatweel=strip_tatweel,
            insert_white_spaces=insert_white_spaces,
            remove_non_digit_repetition=remove_non_digit_repetition,
            replace_slash_with_dash=replace_slash_with_dash,
            map_hindi_numbers_to_arabic=map_hindi_numbers_to_arabic,
            apply_farasa_segmentation=apply_farasa_segmentation,
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        text = re.sub("(URL\s*)+", " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        text = re.sub("(USER\s*)+", " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        # text = re.sub("\B\\[Uu]\w+", "", text)
        text = text.replace("\\U0001f97a", "🥺")
        text = text.replace("\\U0001f928", "🤨")
        text = text.replace("\\U0001f9d8", "😀")
        text = text.replace("\\U0001f975", "😥")
        text = text.replace("\\U0001f92f", "😲")
        text = text.replace("\\U0001f92d", "🤭")
        text = text.replace("\\U0001f9d1", "😐")
        text = text.replace("\\U000e0067", "")
        text = text.replace("\\U000e006e", "")
        text = text.replace("\\U0001f90d", "♥")
        text = text.replace("\\U0001f973", "🎉")
        text = text.replace("\\U0001fa79", "")
        text = text.replace("\\U0001f92b", "🤐")
        text = text.replace("\\U0001f9da", "🦋")
        text = text.replace("\\U0001f90e", "♥")
        text = text.replace("\\U0001f9d0", "🧐")
        text = text.replace("\\U0001f9cf", "")
        text = text.replace("\\U0001f92c", "😠")
        text = text.replace("\\U0001f9f8", "😸")
        text = text.replace("\\U0001f9b6", "💩")
        text = text.replace("\\U0001f932", "🤲")
        text = text.replace("\\U0001f9e1", "🧡")
        text = text.replace("\\U0001f974", "☹")
        text = text.replace("\\U0001f91f", "")
        text = text.replace("\\U0001f9fb", "💩")
        text = text.replace("\\U0001f92a", "🤪")
        text = text.replace("\\U0001f9fc", "")
        text = text.replace("\\U000e0065", "")
        text = text.replace("\\U0001f92e", "💩")
        text = text.replace("\\U000e007f", "")
        text = text.replace("\\U0001f970", "🥰")
        text = text.replace("\\U0001f929", "🤩")
        text = text.replace("\\U0001f6f9", "")
        text = text.replace("🤍", "♥")
        text = text.replace("🦠", "😷")
        text = text.replace("🤢", "مقرف")
        text = text.replace("🤮", "مقرف")
        text = text.replace("🕠", "⌚")
        text = text.replace("🤬", "😠")
        text = text.replace("🤧", "😷")
        text = text.replace("🥳", "🎉")
        text = text.replace("🥵", "🔥")
        text = text.replace("🥴", "☹")
        text = text.replace("🤫", "🤐")
        text = text.replace("🤥", "كذاب")
        text = text.replace("\\u200d", " ")
        text = text.replace("u200d", " ")
        text = text.replace("\\u200c", " ")
        text = text.replace("u200c", " ")
        text = text.replace('"', "'")
        text = text.replace("\\xa0", "")
        text = text.replace("\\u2066", " ")
        text = re.sub("\B\\\[Uu]\w+", "", text)
        text = super(NewArabicPreprocessorBalanced, self).preprocess(text)

        text = " ".join(text.split())
        return text


"""CNNMarbertArabicPreprocessor"""
# ASAD-CNN_MARBERT
class CNNMarbertArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
        remove_elongations=True,
    ):
        if "UBC-NLP" in model_name or "CAMeL-Lab" in model_name:
            keep_emojis = True
            remove_elongations = False
        super().__init__(
            model_name,
            keep_emojis,
            remove_html_markup,
            replace_urls_emails_mentions,
            remove_elongations,
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        text = re.sub("(URL\s*)+", " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        text = re.sub("(USER\s*)+", " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(CNNMarbertArabicPreprocessor, self).preprocess(text)
        text = text.replace("\u200d", " ")
        text = text.replace("u200d", " ")
        text = text.replace("\u200c", " ")
        text = text.replace("u200c", " ")
        text = text.replace('"', "'")
        # text = re.sub('[\d\.]+', ' NUM ', text)
        # text = re.sub('(NUM\s*)+', ' NUM ', text)
        text = multiple_char_pattern.sub(r"\1\1", text)
        text = " ".join(text.split())
        return text


"""Trial5ArabicPreprocessor"""


class Trial5ArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(Trial5ArabicPreprocessor, self).preprocess(text)
        # text = text.replace("السلام عليكم"," ")
        # text = text.replace(find_near_matches("السلام عليكم",text,max_deletions=3,max_l_dist=3)[0].matched," ")
        return text


"""SarcasmArabicPreprocessor"""


class SarcasmArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)
        else:
            return super(SarcasmArabicPreprocessor, self).preprocess(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = text.replace('"', " ")
        text = " ".join(text.split())
        text = super(SarcasmArabicPreprocessor, self).preprocess(text)
        return text


"""NoAOAArabicPreprocessor"""


class NoAOAArabicPreprocessor(ArabertPreprocessor):
    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        if "UBC-NLP" in model_name:
            keep_emojis = True
        super().__init__(
            model_name, keep_emojis, remove_html_markup, replace_urls_emails_mentions
        )
        self.true_model_name = model_name

    def preprocess(self, text):
        if "UBC-NLP" in self.true_model_name:
            return self.ubc_prep(text)
        else:
            return super(NoAOAArabicPreprocessor, self).preprocess(text)

    def ubc_prep(self, text):
        text = re.sub("\s", " ", text)
        text = text.replace("\\n", " ")
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        # replace all possible URLs
        for reg in url_regexes:
            text = re.sub(reg, " URL ", text)
        # replace mentions with USER
        text = re.sub(user_mention_regex, " USER ", text)
        # replace hashtags with HASHTAG
        # text = re.sub(r"#[\w\d]+", " HASH TAG ", text)
        text = text.replace("#", " HASH TAG ")
        text = text.replace("_", " ")
        text = " ".join(text.split())
        text = super(NoAOAArabicPreprocessor, self).preprocess(text)
        text = text.replace("السلام عليكم", " ")
        text = text.replace("ورحمة الله وبركاته", " ")
        matched = find_near_matches("السلام عليكم", text, max_deletions=3, max_l_dist=3)
        if len(matched) > 0:
            text = text.replace(matched[0].matched, " ")
        matched = find_near_matches(
            "ورحمة الله وبركاته", text, max_deletions=3, max_l_dist=3
        )
        if len(matched) > 0:
            text = text.replace(matched[0].matched, " ")
        return text


class CnnBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = 32
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(4, num_filters, (K, config.hidden_size)) for K in filter_sizes]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x = outputs[2][-4:]

        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=outputs.attentions,
        )


class CNNTextClassificationPipeline:
    def __init__(self, model_path, device, return_all_scores=False):
        self.model_path = model_path
        self.model = CnnBertForSequenceClassification.from_pretrained(self.model_path)
        # Special handling
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.return_all_scores = return_all_scores

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.
        Returns:
            Context manager
        Examples::
            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.
        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {
            name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }

    def __call__(self, text):
        """
        Classify the text(s) given as inputs.
        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.
        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:
            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.
            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        # outputs = super().__call__(*args, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs = self.ensure_tensor_on_device(**inputs)
            predictions = self.model(**inputs)[0].cpu()

        predictions = predictions.numpy()

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-predictions))
        else:
            scores = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [
                    {"label": self.model.config.id2label[i], "score": score.item()}
                    for i, score in enumerate(item)
                ]
                for item in scores
            ]
        else:
            return [
                {"label": self.inv_label_map[item.argmax()], "score": item.max().item()}
                for item in scores
            ]
