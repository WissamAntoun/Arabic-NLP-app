import streamlit as st
import awesome_streamlit as ast


def write():
    st.markdown(
        """
    # Arabic Natural Language Processing


    In this HuggingFace space you will be able to test the different Arabic NLP models that my colleges at [AUB MIND Lab](https://sites.aub.edu.lb/mindlab/) have built, with some other applications.

    Check the **Navigation bar** to access the apps:
    - Arabic Text Preprocessor: Test how text imput is treated by our preprocessor
    - Arabic Language Generation: Generate Arabic text using our AraGPT2 language models
    - Arabic Sentiment Analysis: Test the senitment analysis model that won the [Arabic Senitment Analysis competition @ KAUST](https://www.kaggle.com/c/arabic-sentiment-analysis-2021-kaust)
    - Arabic Sarcasm Detection: Test MARBERT trained for sarcasm detection
    - Arabic Question Answering: Test our AraELECTRA QA capabilities
    """
    )
    st.markdown("#")
    col1, col2, col3 = st.beta_columns(3)

    col1.write("## **AraBERT**")
    col1.image("images/arabert_logo.png", width=200)

    col2.write("## **AraGPT2**")
    col2.image("images/AraGPT2.png", width=200)

    col3.write("## **AraElectra**")
    col3.image("images/AraELECTRA.png", width=200)

    st.markdown(
        """

        You can find the more details in the source code and paper linked in our repository on GitHub [repo](https://github.com/aub-mind/arabert).

        ## Dataset

        The pretraining data used for the new **AraBERT** model is also used for **AraGPT2 and AraELECTRA**.

        The dataset consists of 77GB or 200,095,961 lines or 8,655,948,860 words or 82,232,988,358 chars (before applying Farasa Segmentation)

        Our large models were train a TPUv3-128 provided by TFRC.

        For the new dataset we added the unshuffled OSCAR corpus, after we thoroughly filter it, to the previous dataset used in AraBERTv1 but with out the websites that we previously crawled:
        - OSCAR unshuffled and filtered.
        - [Arabic Wikipedia dump](https://archive.org/details/arwiki-20190201) from 2020/09/01
        - [The 1.5B words Arabic Corpus](https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4)
        - [The OSIAN Corpus](https://www.aclweb.org/anthology/W19-4619)
        - Assafir news articles. Huge thank you for Assafir for the data

        ## Models

        Model | HuggingFace Model Name | Size (MB/Params)| Pre-Segmentation |  Hardware | Sequence Length | Batch Size | Num of Steps | Total Time (in Days) |
        ---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
        AraBERTv0.2-base | [bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02) | 543MB / 136M | No | TPUv3-8 | 128 /512 | 2560/384 | 1M/ 2M | 36 |
        AraBERTv0.2-large| [bert-large-arabertv02](https://huggingface.co/aubmindlab/bert-large-arabertv02) | 1.38G / 371M | No | TPUv3-128 | 128 /512 | 13440 / 2056 | 250K / 300K | 7 |
        AraBERTv2-base| [bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2) | 543MB / 136M | Yes | TPUv3-8 |128 /512 | 2560 / 384 | 1M / 2M | 36 |
        AraBERTv2-large| [bert-large-arabertv2](https://huggingface.co/aubmindlab/bert-large-arabertv2) | 1.38G / 371M | Yes | TPUv3-128 |128 /512 | 13440 / 2056|  250K / 300K | 7 |
        AraBERTv0.1-base| [bert-base-arabertv01](https://huggingface.co/aubmindlab/bert-base-arabertv01) | 543MB / 136M | No | TPUv2-8 |128 /512 |128 / 512 | 900K / 300K| 4 |
        AraBERTv1-base| [bert-base-arabert](https://huggingface.co/aubmindlab/bert-base-arabert) | 543MB / 136M | Yes | TPUv2-8 |128 /512 |128 / 512 | 900K / 300K| 4 |
        AraGPT2-base | [aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base) | 527MB/135M | No | TPUv3-128 | 9.7M | 1792 | 125K | 1.5 |
        AraGPT2-medium | [aragpt2-medium](https://huggingface.co/aubmindlab/aragpt2-medium) |  1.38G/370M  | No |TPUv3-8 | 9.7M | 80 | 1M | 15 |
        AraGPT2-large | [aragpt2-large](https://huggingface.co/aubmindlab/aragpt2-large) |  2.98GB/792M  | No |TPUv3-128 | 9.7M | 256 | 220k | 3 |
        AraGPT2-mega | [aragpt2-mega](https://huggingface.co/aubmindlab/aragpt2-mega) |  5.5GB/1.46B  |No |TPUv3-128 | 9.7M | 256 | 800K | 9 |
        AraELECTRA-base-generator | [araelectra-base-generator](https://huggingface.co/aubmindlab/araelectra-base-generator) |  227MB/60M  | No | TPUv3-8 | 512 | 256 | 2M | 24
        AraELECTRA-base-discriminator | [araelectra-base-discriminator](https://huggingface.co/aubmindlab/araelectra-base-discriminator) |  516MB/135M  | No | TPUv3-8 | 512 | 256 | 2M | 24

        All models are available in the `HuggingFace` model page under the [aubmindlab](https://huggingface.co/aubmindlab/) name. Checkpoints are available in PyTorch, TF2 and TF1 formats.

        # Preprocessing

        You can test the Arabic Preprocessing pipeline in the Arabic Text Preprocessing page.

        It is recommended to apply our preprocessing function before training/testing on any dataset.
        **Install farasapy to segment text for AraBERT v1 & v2 `pip install farasapy`**

        ```python
        from arabert.preprocess import ArabertPreprocessor

        model_name = "aubmindlab/bert-base-arabertv2"
        arabert_prep = ArabertPreprocessor(model_name=model_name)

        text = "ولن نبالغ إذا قلنا: إن 'هاتف' أو 'كمبيوتر المكتب' في زمننا هذا ضروري"
        arabert_prep.preprocess(text)
        >>>"و+ لن نبالغ إذا قل +نا : إن ' هاتف ' أو ' كمبيوتر ال+ مكتب ' في زمن +نا هذا ضروري"
        ```

        You can also use the `unpreprocess()` function to reverse the preprocessing changes, by fixing the spacing around non alphabetical characters, and also de-segmenting if the model selected need pre-segmentation. We highly recommend unprocessing generated content of `AraGPT2` model, to make it look more natural.
        ```python
        output_text = "و+ لن نبالغ إذا قل +نا : إن ' هاتف ' أو ' كمبيوتر ال+ مكتب ' في زمن +نا هذا ضروري"
        arabert_prep.unpreprocess(output_text)
        >>>"ولن نبالغ إذا قلنا: إن 'هاتف' أو 'كمبيوتر المكتب' في زمننا هذا ضروري"
        ```

        # If you used this model please cite us as :

        ## AraBERT
        Google Scholar has our Bibtex wrong (missing name), use this instead
        ```
        @inproceedings{antoun2020arabert,
        title={AraBERT: Transformer-based Model for Arabic Language Understanding},
        author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
        booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
        pages={9}
        }
        ```
        ## AraGPT2
        ```
        @inproceedings{antoun-etal-2021-aragpt2,
            title = "{A}ra{GPT}2: Pre-Trained Transformer for {A}rabic Language Generation",
            author = "Antoun, Wissam  and
            Baly, Fady  and
            Hajj, Hazem",
            booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
            month = apr,
            year = "2021",
            address = "Kyiv, Ukraine (Virtual)",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/2021.wanlp-1.21",
            pages = "196--207",
        }
        ```

        ## AraELECTRA
        ```
        @inproceedings{antoun-etal-2021-araelectra,
            title = "{A}ra{ELECTRA}: Pre-Training Text Discriminators for {A}rabic Language Understanding",
            author = "Antoun, Wissam  and
            Baly, Fady  and
            Hajj, Hazem",
            booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
            month = apr,
            year = "2021",
            address = "Kyiv, Ukraine (Virtual)",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/2021.wanlp-1.20",
            pages = "191--195",
        }
        ```


        # Acknowledgments
        Thanks to TensorFlow Research Cloud (TFRC) for the free access to Cloud TPUs, couldn't have done it without this program, and to the [AUB MIND Lab](https://sites.aub.edu.lb/mindlab/) Members for the continous support. Also thanks to [Yakshof](https://www.yakshof.com/#/) and Assafir for data and storage access. Another thanks for Habib Rahal (https://www.behance.net/rahalhabib), for putting a face to AraBERT.

        # Contacts
        **Wissam Antoun**: [Linkedin](https://www.linkedin.com/in/wissam-antoun-622142b4/) | [Twitter](https://twitter.com/wissam_antoun) | [Github](https://github.com/WissamAntoun) | wfa07 (AT) mail (DOT) aub (DOT) edu | wissam.antoun (AT) gmail (DOT) com

        **Fady Baly**: [Linkedin](https://www.linkedin.com/in/fadybaly/) | [Twitter](https://twitter.com/fadybaly) | [Github](https://github.com/fadybaly) | fgb06 (AT) mail (DOT) aub (DOT) edu | baly.fady (AT) gmail (DOT) com

        """
    )
