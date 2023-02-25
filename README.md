# Awesome Transformers

![Transformers](logo.png 'MarineGEO logo')

A curated list of awesome transformer models.

If you want to contribute to this list, send a pull request or reach out to me on twitter: [@abacaj](https://twitter.com/abacaj). Let's make this list useful.

There are a number of models available that are not entirely open source (non-commercial, etc), this repository should serve to also make you aware of that. Tracking the original source/company of the model will help.

I would also eventually like to add model use cases. So it is easier for others to find the right one to fine-tune.

_Format_:

- Model name: short description, usually from paper
  - Model link (usually huggingface or github)
  - Paper link
  - Source as company or group
  - Model license

## Table of Contents

- [Encoder (autoencoder) models](#encoder)
  - [ALBERT](#albert)
  - [BERT](#bert)
  - [DistilBERT](#distilbert)
  - [Electra](#electra)
  - [RoBERTa](#roberta)
- [Decoder (autoregressive) models](#encoder)
  - [BioGPT](#bio-gpt)
  - [CodeGen](#codegen)
  - [GPT](#gpt)
  - [GPT-2](#gpt-2)
  - [GPT-J](#gpt-j)
  - [GPT-NEO](#gpt-neo)
  - [GPT-NEOX](#gpt-neox)
  - [NeMo Megatron-GPT](#nemo)
  - [OPT](#opt)
  - [BLOOM](#bloom)
  - [GLM](#glm)
  - [YaLM](#yalm)
- [Encoder+decoder (seq2seq) models](#encoder-decoder)
  - [T5](#t5)
  - [FLAN-T5](#flan-t5)
  - [Code-T5](#code-t5)
  - [Bart](#bart)
  - [Pegasus](#pegasus)
  - [MT5](#mt5)
- [Multimodal models](#multimodal)
  - [Donut](#donut)
- [Vision models](#vision)

<a name="encoder"></a>

## Encoder models

<a name="albert"></a>

- ALBERT: "A Lite" version of BERT
  - [Model](https://huggingface.co/models?other=albert)
  - [Paper](https://arxiv.org/pdf/1909.11942.pdf)
  - Google
  - Apache v2
    <a name="bert"></a>
- BERT: Bidirectional Encoder Representations from Transformers
  - [Model](https://huggingface.co/models?other=bert)
  - [Paper](https://arxiv.org/pdf/1810.04805.pdf)
  - Google
  - Apache v2
    <a name="distilbert"></a>
- DistilBERT: Distilled version of BERT smaller, faster, cheaper and lighter
  - [Model](https://huggingface.co/models?other=distilbert)
  - [Paper](https://arxiv.org/pdf/1910.01108.pdf)
  - HuggingFace
  - Apache v2
    <a name="electra"></a>
- Electra: Pre-training Text Encoders as Discriminators Rather Than Generators
  - [Model](https://huggingface.co/models?other=electra)
  - [Paper](https://arxiv.org/pdf/2003.10555.pdf)
  - Google
  - Apache v2
    <a name="roberta"></a>
- RoBERTa: Robustly Optimized BERT Pretraining Approach
  - [Model](https://huggingface.co/models?other=roberta)
  - [Paper](https://arxiv.org/pdf/1907.11692.pdf)
  - Facebook
  - MIT

<a name="decoder"></a>

## Decoder models

<a name="bio-gpt"></a>

- BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining
  - [Model](https://huggingface.co/microsoft/biogpt)
  - [Paper](https://arxiv.org/pdf/2210.10341.pdf)
  - Microsoft
  - MIT
- CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis
  - [Model](https://huggingface.co/models?sort=downloads&search=salesforce%2Fcodegen)
  - [Paper](https://arxiv.org/pdf/2203.13474.pdf)
  - Salesforce
  - BSD 3-Clause
    <a name="CodeGen"></a>
- GPT: Bidirectional Encoder Representations from Transformers
  - [Model](https://huggingface.co/openai-gpt)
  - [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - OpenAI
  - MIT
    <a name="gpt"></a>
- GPT-2: Distilled version of BERT smaller, faster, cheaper and lighter
  - [Model](https://huggingface.co/models?search=gpt-2)
  - [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - OpenAI
  - MIT
    <a name="gpt-2"></a>
- GPT-J: GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model
  - [Model](https://huggingface.co/EleutherAI/gpt-j-6B)
  - [Paper]()
  - EleutherAI
  - Apache v2
    <a name="gpt-j"></a>
- GPT-NEO:
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="gpt-neo"></a>
- NeMo Megatron-GPT:
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="nemo"></a>
- OPT:
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="opt"></a>
- BLOOM:
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="bloom"></a>
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="glm"></a>
- YaLM:
  - [Model]()
  - [Paper]()
  - .
  - .
    <a name="yalm"></a>

<a name="encoder-decoder"></a>

## Encoder+decoder (seq2seq) models

<a name="bio-gpt"></a>

- T5:
  - [Model]()
  - [Paper]()
  -
  -
- FLAN-T5:
  - [Model]()
  - [Paper]()
  -
  -
- Code-T5:
  - [Model]()
  - [Paper]()
  -
  -
- Pegasus:
  - [Model]()
  - [Paper]()
  -
  -
- MT5:
  - [Model]()
  - [Paper]()
  -
  -
