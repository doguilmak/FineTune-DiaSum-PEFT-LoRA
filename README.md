<p  align="center">
<img src="https://raw.githubusercontent.com/doguilmak/FineTune-DiaSum-PEFT-LoRA/main/assets/pexels-googledeepmind-18069696.jpg" height=450 width=2000 alt="Cover">
</p>  

<small>Picture Source: <a  href="https://www.pexels.com/@googledeepmind/">Google Deepmind Pexels</a></small>

<br>

<br>

## Abstract
In this project, we explore the use of Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) to fine-tune large pre-trained language models for the task of dialogue summarization. These techniques allow for efficient adaptation of pre-trained models to new tasks with reduced computational resources, making them accessible and practical for a wider range of applications.

<br>

## Introduction
Large pre-trained language models have shown remarkable performance across various natural language processing (NLP) tasks. However, fine-tuning these models for specific tasks can be resource-intensive. Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) are techniques that address this challenge by updating only a subset of the model parameters. This project demonstrates the application of PEFT and LoRA to fine-tune a pre-trained language model for dialogue summarization.

<br>

## Methodology

The main steps involved in this process are:

1. **Defining the Configuration**:
   - Setting up the `LoraConfig` to specify the parameters for low-rank adaptation.

2. **Loading the Model and Tokenizer**:
   - Using the `AutoModelForSeq2SeqLM` and `AutoTokenizer` from the Hugging Face library.

3. **Applying PEFT to the Model**:
   - Integrating the PEFT configuration with the pre-trained model.

4. **Data Preparation**:
   - Preparing the dataset for dialogue summarization and defining data collators.

5. **Training**:
   - Fine-tuning the model using the `Seq2SeqTrainer` from the Hugging Face library.

6. **Evaluation**:
   - Assessing the performance of the fine-tuned model on a validation set.

<br>

Here you can find the **[IPython Notebook](https://github.com/doguilmak/FineTune-DiaSum-PEFT-LoRA/blob/main/DiaSum_PEFT_LoRA.ipynb)** file to do all these step by step.

<br>

## PEFT (Parameter-Efficient Fine-Tuning)

PEFT is a technique designed to fine-tune pre-trained models on specific tasks using fewer parameters and computational resources. The key idea is to modify only a small subset of the model parameters, keeping the majority of the pre-trained model fixed. This approach reduces memory and computational requirements, making it feasible to adapt large models to new tasks with limited resources.

### Key Benefits of PEFT:
1. **Reduced Computational Resources**: Fine-tuning only a small portion of the model parameters requires less memory and computational power.
2. **Faster Training**: Updating fewer parameters speeds up the training process.
3. **Effective for Large Models**: Useful for adapting large pre-trained models to new tasks without extensive computational infrastructure.

<br>

## LoRA (Low-Rank Adaptation)

LoRA introduces a low-rank decomposition to the model parameters being fine-tuned. Instead of updating the full weight matrices, LoRA updates a low-rank approximation, significantly reducing the number of trainable parameters.

### How LoRA Works:
1. **Low-Rank Decomposition**: Decomposes weight matrices into two smaller matrices with a lower rank, reducing the number of parameters to be updated.
2. **Injecting Low-Rank Updates**: During training, only the low-rank matrices are updated while the original pre-trained weights remain fixed.
3. **Efficient Adaptation**: Updates only a low-rank approximation, allowing the model to adapt to new tasks efficiently without compromising performance.

### Key Benefits of LoRA:
1. **Parameter Efficiency**: Significantly reduces the number of trainable parameters.
2. **Scalability**: Can be applied to very large models.
3. **Maintaining Performance**: Despite fewer trainable parameters, maintains competitive performance.

<br>

## Configuration Details

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank decomposition. Controls the size of low-rank matrices.
    lora_alpha=32,  # Scaling factor for the low-rank matrices. Balances the contribution of the low-rank adaptation.
    lora_dropout=0.05,  # Dropout rate applied to the low-rank adaptation. Prevents overfitting by randomly dropping some adaptations.
    bias="none",  # Type of bias adjustment. "none" indicates no bias terms are used in the low-rank adaptation.
    task_type=TaskType.SEQ_2_SEQ_LM  # Task type for the model. Here, it's set for sequence-to-sequence language modeling.
)
```
<br>

## Conclusion

This project demonstrates the effectiveness of PEFT and LoRA techniques for fine-tuning pre-trained language models on dialogue summarization tasks. By reducing the computational resources required for fine-tuning, these methods make it feasible to adapt large models to new tasks, offering practical solutions for a wider range of applications.

<br>

## Contact Me

If you have something to say to me please contact me:

*	Twitter: [Doguilmak](https://twitter.com/Doguilmak)
*	Mail address: doguilmak@gmail.com
