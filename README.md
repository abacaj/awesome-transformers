# Awesome Transformers

![Transformers](logo.png 'MarineGEO logo')

A curated list of awesome transformer models.

If you want to contribute to this list, send a pull request or reach out to me on twitter: [@abacaj](https://twitter.com/abacaj). Let's make this list useful.

There are a number of models available that are not entirely open source (non-commercial, etc), this repository should serve to also make you aware of that. Tracking the original source/company of the model also helps.

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
  - [CodeGen](#gpt)
  - [GPT](#gpt)
  - [GPT-2](#gpt)
  - [GPT-J](#gpt)
  - [GPT-NEO](#gpt)
  - [GPT-NEOX](#gpt)
  - [NeMo Megatron-GPT](#gpt)
  - [OPT](#gpt)
  - [BLOOM](#gpt)
  - [GLM](#gpt)
  - [YaLM](#gpt)
- [Encoder+decoder (seq2seq) models](#encoder-decoder)
  - [T5](#t5)
  - [FLAN-T5](#t5)
  - [Bart](#t5)
  - [Pegasus](#t5)
  - [MT5](#t5)
- [Multimodal models](#multimodal)
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

## Decoder models

<a name="bio-gpt"></a>

- BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining
  - [Model](https://huggingface.co/microsoft/biogpt)
  - [Paper](https://arxiv.org/pdf/2210.10341.pdf)
  - Microsoft
  - MIT
