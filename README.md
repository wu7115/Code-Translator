# QLoRA Fine-Tuned Code Translator (C++ → Python)

This repository contains a QLoRA-fine-tuned LLaMA 3.1 8B model that I trained to translate C++ code into Python with strong syntactic and dataflow understanding. Built for efficient training and robust code translation, my model outperforms GPT-4o-mini in multiple CodeBLEU metrics.

## Project Highlights

- **Base Model:** [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- **Fine-tuning Method:** QLoRA (Quantized Low-Rank Adapter)
- **Quantization:** 4-bit NF4 with double quantization
- **Evaluation Metric:** CodeBLEU with syntax, dataflow, and weighted n-gram breakdown
- **Beats GPT-4o-mini** on custom test dataset in structured code translation

---

## Dataset

I used the [**XLCoST**](https://github.com/reddy-lab-code-research/XLCoST) dataset:  
**"XLCoST: A Benchmark Dataset for Cross-lingual Code Intelligence"**, which provides parallel code examples across multiple programming languages including C++ and Python.

From XLCoST, I extracted line-mapped C++ and Python function pairs, tokenized and formatted them to suit a prompt-based instruction fine-tuning approach.

---

## Training Details

| Feature                  | Value                                 |
|--------------------------|---------------------------------------|
| Model                    | Meta-LLaMA 3.1 8B                     |
| Finetuning Strategy      | QLoRA + LoRA                          |
| Quantization             | 4-bit NF4                             |
| Optimizer                | `paged_adamw_32bit`                   |
| Scheduler                | Cosine LR Scheduler                   |
| Max Seq Length           | 512                                   |
| Epochs                   | 1                                     |
| Dataset Format           | Line-mapped C++ and Python files      |
| Evaluation Metric        | CodeBLEU (0.7.0)                      |
| Validation Sample Size   | 100 examples                          |

---

## Evaluation

| Metric                   | My Fine-Tuned Model    | GPT-4o-mini    |
|--------------------------|------------------------|----------------|
| CodeBLEU                 | **61.78%**             | 29.41%         |
| N-gram Match             | 9.94%                  | 4.21%          |
| Weighted N-gram Match    | **75.29%**             | 7.53%          |
| Syntax Match             | **79.51%**             | 48.68%         |
| Dataflow Match           | **85.37%**             | 57.22%         |
| Exact Match Accuracy     | 0.00%                  | 0.00%          |

> 🔍 The model shows strong generalization in structure-aware metrics like syntax and dataflow, even without exact textual match.

---

## Dataset Format

Custom tokenized files in line-by-line format:
- `train-C++-Python-tok.cpp`
- `train-C++-Python-tok.py`
- `val-C++-Python-tok.cpp`
- `val-C++-Python-tok.py`
- `test-C++-Python-tok.cpp`
- `test-C++-Python-tok.py`

Each line contains one C++ or Python sample, paired by line index.  
Preprocessing includes removing structural tokens and cleaning indentation.

---

## Training

```python
# Load training pairs
train = load_code_pair("train-C++-Python-tok.cpp", "train-C++-Python-tok.py")
val = load_code_pair("val-C++-Python-tok.cpp", "val-C++-Python-tok.py")

# Launch training
trainer = SFTTrainer(...)
trainer.train()

# Push to Hugging Face
trainer.model.push_to_hub("my-model-name")

## Model Access

My fine-tuned model is hosted on Hugging Face:

[https://huggingface.co/wu7115/code-translator-2025-04-28_18.24.41](https://huggingface.co/wu7115/code-translator-2025-04-28_18.24.41)
