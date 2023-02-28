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
  - [DeBERTaV3](#debertav3)
  - [Electra](#electra)
  - [RoBERTa](#roberta)
- [Decoder (autoregressive) models](#decoder)
  - [BioGPT](#bio-gpt)
  - [CodeGen](#codegen)
  - [LLaMa](#llama)
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
  - [UL2](#ul2)
  - [EdgeFormer](#edgeformer)
- [Multimodal models](#multimodal)
  - [Donut](#donut)
  - [LayoutLMv3](#layoutlmv3)
  - [TrOCR](#trocr)
  - [CLIP](#clip)
  - [Unified-IO](#unifiedio)
- [Vision models](#vision)
  - [DiT](#dit)
  - [DETR](#detr)
  - [EfficientFormer](#efficientformer)
- [Audio models](#audio)
  - [Whisper](#whisper)
  - [VALL-E](#valle)
- [Recommendation models](#recommendation)
  - [P5](#p5)

<a name="encoder"></a>

## Encoder models

<a name="albert"></a>

- ALBERT: "A Lite" version of BERT
  - [Model](https://huggingface.co/models?other=albert)
  - [Paper](https://arxiv.org/pdf/1909.11942.pdf)
  - Google
  - Apache v2
- BERT: Bidirectional Encoder Representations from Transformers
  <a name="bert"></a>
  - [Model](https://huggingface.co/models?other=bert)
  - [Paper](https://arxiv.org/pdf/1810.04805.pdf)
  - Google
  - Apache v2
- DistilBERT: Distilled version of BERT smaller, faster, cheaper and lighter
  <a name="distilbert"></a>
  - [Model](https://huggingface.co/models?other=distilbert)
  - [Paper](https://arxiv.org/pdf/1910.01108.pdf)
  - HuggingFace
  - Apache v2
- DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
  <a name="debertav3"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=microsoft%2Fdeberta-v3)
  - [Paper](https://arxiv.org/pdf/2111.09543.pdf)
  - Microsoft
  - MIT
- Electra: Pre-training Text Encoders as Discriminators Rather Than Generators
  <a name="electra"></a>
  - [Model](https://huggingface.co/models?other=electra)
  - [Paper](https://arxiv.org/pdf/2003.10555.pdf)
  - Google
  - Apache v2
- RoBERTa: Robustly Optimized BERT Pretraining Approach
  <a name="roberta"></a>
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
  <a name="codegen"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=salesforce%2Fcodegen)
  - [Paper](https://arxiv.org/pdf/2203.13474.pdf)
  - Salesforce
  - BSD 3-Clause
- LLaMa: Open and Efficient Foundation Language Models
  <a name="llama"></a>
  - [Model](https://github.com/facebookresearch/llama)
  - [Paper](https://research.facebook.com/file/1574548786327032/LLaMA--Open-and-Efficient-Foundation-Language-Models.pdf)
  - Facebook
  - [Requires approval, non-commercial](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- GPT: Improving Language Understanding by Generative Pre-Training
  <a name="gpt"></a>
  - [Model](https://huggingface.co/openai-gpt)
  - [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - OpenAI
  - MIT
- GPT-2: Language Models are Unsupervised Multitask Learners
  <a name="gpt-2"></a>
  - [Model](https://huggingface.co/models?search=gpt-2)
  - [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - OpenAI
  - MIT
- GPT-J: A 6 Billion Parameter Autoregressive Language Model
  <a name="gpt-j"></a>
  - [Model](https://huggingface.co/EleutherAI/gpt-j-6B)
  - [Paper](https://github.com/kingoflolz/mesh-transformer-jax)
  - EleutherAI
  - Apache v2
- GPT-NEO: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow
  <a name="gpt-neo"></a>
  - [Model](https://huggingface.co/models?search=gpt-neo)
  - [Paper](https://doi.org/10.5281/zenodo.5297715)
  - EleutherAI
  - MIT
- GPT-NEOX-20B: An Open-Source Autoregressive Language Model
  <a name="gpt-neox"></a>
  - [Model](https://huggingface.co/EleutherAI/gpt-neox-20b)
  - [Paper](https://arxiv.org/pdf/2204.06745.pdf)
  - EleutherAI
  - Apache v2
- NeMo Megatron-GPT: Megatron-GPT 20B is a transformer-based language model.
  <a name="nemo"></a>
  - [Model](https://huggingface.co/nvidia/nemo-megatron-gpt-20B)
  - [Paper](https://arxiv.org/pdf/1909.08053.pdf)
  - NVidia
  - CC BY 4.0
- OPT: Open Pre-trained Transformer Language Models
  <a name="opt"></a>
  - [Model](https://huggingface.co/models?search=facebook%2Fopt)
  - [Paper](https://arxiv.org/pdf/2205.01068.pdf?fbclid=IwAR1Fhxr_i3UK3ttigVDGBwbtO-3zLzjTwnyn0dkYt8rf6hxUAUS7Sk7VrYk)
  - Facebook
  - [Requires approval, non-commercial](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md?fbclid=IwAR2jiCf2R9fTouGGF7v8Tt7Yq8sSVOMot0YIE8ibaP9b2avxw2bEbEaTJZY)
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model
  <a name="bloom"></a>
  - [Model](https://huggingface.co/bigscience/bloom)
  - [Paper](https://arxiv.org/pdf/2211.05100.pdf)
  - BigScience
  - [OpenRAIL, use-based restrictions](https://huggingface.co/spaces/bigscience/license)
- GLM: An Open Bilingual Pre-Trained Model
  <a name="glm"></a>
  - [Model](https://github.com/THUDM/GLM-130B)
  - [Paper](https://arxiv.org/pdf/2210.02414.pdf)
  - Knowledge Engineering Group (KEG) & Data Mining at Tsinghua University
  - [Custom license, see restrictions](https://github.com/THUDM/GLM-130B/blob/main/MODEL_LICENSE)
- YaLM: Pretrained language model with 100B parameters
  <a name="yalm"></a>
  - [Model](https://github.com/yandex/YaLM-100B)
  - [Paper](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6)
  - Yandex
  - Apache v2

<a name="encoder-decoder"></a>

## Encoder+decoder (seq2seq) models

<a name="bio-gpt"></a>

- T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
  <a name="t5"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=t5)
  - [Paper](https://arxiv.org/pdf/1910.10683.pdf)
  - Google
  - Apache v2
- FLAN-T5: Scaling Instruction-Finetuned Language Models
  <a name="flan-t5"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=flan-t5)
  - [Paper](https://arxiv.org/pdf/2210.11416.pdf)
  - Google
  - Apache v2
- Code-T5: Identifier-aware Unified Pre-trained Encoder-Decoder Models
  for Code Understanding and Generation
  <a name="code-t5"></a>
  - [Model](https://huggingface.co/models?search=code-t5)
  - [Paper](https://arxiv.org/pdf/2109.00859.pdf)
  - Salesforce
  - BSD 3-Clause
- Bart: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
  <a name="bart"></a>
  - [Model](https://huggingface.co/facebook/bart-large)
  - [Paper](https://arxiv.org/pdf/1910.13461.pdf)
  - Facebook
  - Apache v2
- Pegasus: Pre-training with Extracted Gap-sentences for Abstractive Summarization
  <a name="pegasus"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=pegasus)
  - [Paper](https://arxiv.org/pdf/1912.08777.pdf)
  - Google
  - Apache v2
- MT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
  <a name="mt5"></a>
  - [Model](https://huggingface.co/models?search=mt5)
  - [Paper](https://arxiv.org/pdf/2010.11934.pdf)
  - Google
  - Apache v2
- UL2: Unifying Language Learning Paradigms
  <a name="ul2"></a>
  - [Model](https://huggingface.co/google/ul2)
  - [Paper](https://arxiv.org/pdf/2205.05131v1.pdf)
  - Google
  - Apache v2
- EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation
  <a name="edgeformer"></a>
  - [Model](https://github.com/microsoft/unilm/tree/master/edgelm)
  - [Paper](https://arxiv.org/pdf/2202.07959.pdf)
  - Microsoft
  - MIT

<a name="multimodal"></a>

## Multimodal models

<a name="donut"></a>

- Donut: OCR-free Document Understanding Transformer
  - [Model](https://huggingface.co/models?sort=downloads&search=clova%2Fdonut)
  - [Paper](https://arxiv.org/pdf/2111.15664.pdf)
  - ClovaAI
  - MIT
- LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
  <a name="layoutlmv3"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=microsoft%2Flayoutlmv3)
  - [Paper](https://arxiv.org/pdf/2204.08387.pdf)
  - Microsoft
  - CC BY-NC-SA 4.0 (non-commercial)
- TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models
  <a name="trocr"></a>
  - [Model](https://huggingface.co/models?search=trocr)
  - [Paper](https://arxiv.org/abs/2109.10282)
  - Microsoft
  - [Inherits MIT license](https://github.com/microsoft/unilm/tree/master/trocr#license)
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
  <a name="clip"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=openai%2Fclip)
  - [Paper](https://arxiv.org/pdf/2103.00020.pdf)
  - OpenAI
  - MIT
- Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks
  <a name="unifiedio"></a>
  - [Model](https://github.com/allenai/unified-io-inference)
  - [Paper](https://arxiv.org/pdf/2206.08916.pdf)
  - allenai
  - Apache v2

<a name="vision"></a>

## Vision models

<a name="dit"></a>

- DiT: Self-supervised Pre-training for Document Image Transformer
  - [Model](https://huggingface.co/models?search=microsoft/dit)
  - [Paper](https://arxiv.org/pdf/2203.02378.pdf)
  - Microsoft
  - [Inherits MIT license](https://github.com/microsoft/unilm/tree/master/dit#license)
- DETR: End-to-End Object Detection with Transformers
  <a name="detr"></a>
  - [Model](https://huggingface.co/models?search=facebook/detr)
  - [Paper](https://arxiv.org/pdf/2005.12872.pdf)
  - Facebook
  - Apache v2
- EfficientFormer: Vision Transformers at MobileNet Speed
  <a name="efficientformer"></a>
  - [Model](https://huggingface.co/models?sort=downloads&search=snap-research%2Fefficientformer)
  - [Paper](https://arxiv.org/pdf/2206.01191.pdf)
  - Snap
  - Apache v2

<a name="audio"></a>

## Audio models

<a name="whisper"></a>

- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
  - [Model](https://huggingface.co/models?sort=downloads&search=openai%2Fwhisper)
  - [Paper](https://arxiv.org/pdf/2212.04356.pdf)
  - OpenAI
  - MIT
- VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers
  <a name="valle"></a>
  - [Model (unofficial)](https://github.com/enhuiz/vall-e)
    - MIT but has a dependency on a CC-BY-NC library
  - [Model (unofficial)](https://github.com/lifeiteng/vall-e)
    - Apache v2
  - [Paper](https://arxiv.org/pdf/2301.02111.pdf)
  - Microsoft

<a name="recommendation"></a>

## Recommendation models

<a name="p5"></a>

- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)
  - [Model](https://github.com/jeykigung/P5)
  - [Paper](https://arxiv.org/abs/2203.13366)
  - Rutgers
  - Apache v2
