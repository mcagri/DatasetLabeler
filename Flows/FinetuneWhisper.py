from transformers import (WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor,
                          WhisperTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer)

from prefect import flow, task, deploy
from datasets import load_from_disk, DatasetDict
import Config.Config as Config
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


metric = evaluate.load("wer")
model_name = 'openai/whisper-medium'
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Turkish", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Turkish", task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@task
def load_dataset(dataset_path: str = "../Dataset") -> DatasetDict:
    dataset = load_from_disk(dataset_path)

    if 'train' not in dataset or 'validation' not in dataset:
        # Split the dataset into train and validation sets (80-20 split)
        dataset = dataset.train_test_split(test_size=0.2)
        dataset = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })
    return dataset


@task
def finetune(dataset, output: str="../Results/whisper-medium-tr"):
    model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="auto", use_cache=False)
    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"

    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=1,
        max_steps=10,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=10,
        eval_steps=10,
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()


@flow(log_prints=True)
def finetune_whisper():
    dataset = load_dataset(Config.dataset_path)
    dataset = dataset.map(prepare_dataset,remove_columns=dataset.column_names["train"],num_proc=1)
    finetune(dataset)


if __name__ == "__main__":
    finetune_whisper.serve(name="finetune_whisper",
                            tags = ["medium", "DataLabeler", "Whisper", "Finetune"],
                            description = "Finetune Whisper and save model",)
