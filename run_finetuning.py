# ==============================================================================
# Phase 1: Import Necessary Libraries
# ==============================================================================
print("Loading libraries...")
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

print("Libraries loaded successfully.")

# ==============================================================================
# Phase 2: Load the Dataset (with the corrected configuration name)
# ==============================================================================
print("\n--- Starting Phase 2: Data Loading ---")
try:
    # *** CORRECTION MADE HERE ***
    # The original attempt used the name "con", which was incorrect.
    # The error message showed that the correct name is "mentalmanip_con".
    # We are now using the correct name to load the specific dataset configuration.
    dataset_name = "audreyeleven/MentalManip"
    config_name = "mentalmanip_con"

    dataset = load_dataset(dataset_name, config_name)
    print(f"Successfully loaded the '{config_name}' configuration from '{dataset_name}'.")

    # The dataset object is a dictionary with train, validation, and test splits
    print("\nDataset structure:")
    print(dataset)

    # Inspect a single example to understand its structure
    print("\nExample from the training set:")
    print(dataset['train'][0])

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    print("Please check the dataset name, configuration, and your internet connection.")
    # Exit the script if the data can't be loaded, as nothing else can be done.
    exit()

# ==============================================================================
# Phase 3: Preprocess the Data
# ==============================================================================
print("\n--- Starting Phase 3: Data Preprocessing ---")
# We'll use a standard BERT model, so we load its corresponding tokenizer.
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"Tokenizer for '{model_checkpoint}' loaded.")


# This function tokenizes the text. BERT models have a maximum token limit (512 for bert-base-uncased).
# `truncation=True` will cut off any text longer than the maximum length.
def preprocess_function(examples):
    return tokenizer(examples["dialogue"], truncation=True)


# Apply the tokenizer to all examples in the train, validation, and test sets.
# `batched=True` processes multiple examples at once for efficiency.
print("Applying tokenizer to the dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Tokenization complete.")

# Rename the 'Manipulation' column to 'labels' as this is the expected column name for the Trainer API.
tokenized_datasets = tokenized_datasets.rename_column("manipulative", "labels")

# We no longer need the original 'Dialogue' text column after tokenization.
tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])

# Set the format to PyTorch tensors so the data can be loaded into the model.
tokenized_datasets.set_format("torch")

print("\nExample from the processed training set (now with input_ids and labels):")
print(tokenized_datasets['train'][0])

# # ==============================================================================
# # Phase 4: Set Up the Fine-Tuning Process
# # ==============================================================================
# print("\n--- Starting Phase 4: Model and Trainer Setup ---")
#
# # Load the pre-trained BERT model with a classification head on top.
# # `num_labels=2` tells the model we are doing binary classification (manipulative vs. not-manipulative).
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
# print(f"Model '{model_checkpoint}' loaded for sequence classification.")
#
# # Define the directory where the trained model and results will be saved.
# output_dir = "bert-mental-manipulation-detector"
#
# # Set up training arguments that control the fine-tuning process.
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=2e-5,  # A standard learning rate for fine-tuning BERT.
#     per_device_train_batch_size=8,  # Batch size per GPU. Reduce if you run out of memory.
#     per_device_eval_batch_size=8,  # Batch size for evaluation.
#     num_train_epochs=3,  # A common number of epochs for fine-tuning tasks.
#     weight_decay=0.01,  # Regularization to prevent overfitting.
#     #made an edit here
#     eval_strategy="epoch",  # Run evaluation at the end of each epoch.
#     save_strategy="epoch",  # Save a checkpoint at the end of each epoch.
#     load_best_model_at_end=True,  # The Trainer will load the best model (based on loss) at the end.
#     metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model.
#     greater_is_better=False,  # For loss, a lower value is better.
# )
#
# # Define the metrics we want to compute during evaluation.
# accuracy_metric = evaluate.load("accuracy")
# f1_metric = evaluate.load("f1")
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#     f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
#
#     return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}
#
#
# # Instantiate the Trainer, which handles the entire training and evaluation loop.
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
# print("Trainer setup complete.")

# ==============================================================================
# Phase 4: Set Up the Fine-Tuning Process (with Validation Split)
# ==============================================================================
print("\n--- Starting Phase 4: Model and Trainer Setup ---")

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
print(f"Model '{model_checkpoint}' loaded for sequence classification.")

output_dir = "bert-mental-manipulation-detector"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    # Using the 'eval_strategy' that worked for you
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


# --- THIS IS THE KEY CORRECTION ---
# Since the dataset only has a 'train' split, we create our own validation set
# by splitting the training data. We'll use 10% of the data for validation.
print("Splitting the training set to create a validation set...")
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)

# Now, train_test_split is a new DatasetDict with 'train' and 'test' keys
# {'train': Dataset(...), 'test': Dataset(...)}
print("Split complete. New dataset structure for training:")
print(train_test_split)

trainer = Trainer(
    model=model,
    args=training_args,
    # Pass the new, smaller training set (90%)
    train_dataset=train_test_split["train"],
    # Pass the new validation set (10%) that we just created
    eval_dataset=train_test_split["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Trainer setup complete.")

# ==============================================================================
# Phase 5: Train the Model
# ==============================================================================
print("\n--- Starting Phase 5: Model Training ---")
# This command starts the fine-tuning. It will display a progress bar.
# If you have a compatible GPU, PyTorch will automatically use it.
try:
    trainer.train()
    print("Fine-tuning complete.")
except torch.cuda.OutOfMemoryError:
    print("\nCUDA Out of Memory Error!")
    print("Try reducing `per_device_train_batch_size` in the TrainingArguments and run the script again.")
    exit()

# # ==============================================================================
# # Phase 6: Evaluate the Final Model
# # ==============================================================================
# print("\n--- Starting Phase 6: Final Evaluation on Test Set ---")
# # The trainer automatically loaded the best model. Now we evaluate it on the unseen test set.
# test_results = trainer.evaluate(tokenized_datasets["test"])
#
# print("\nTest Set Evaluation Results:")
# print(test_results)
#
# # ==============================================================================
# # Phase 7: Use the Model for Inference
# # ==============================================================================
# print("\n--- Starting Phase 7: Inference with New Text ---")
# # Save the final, fine-tuned model and tokenizer.
# final_model_path = f"./{output_dir}/final"
# trainer.save_model(final_model_path)
# print(f"Final model saved to {final_model_path}")
#
# # Load the fine-tuned model into a pipeline for easy prediction.
# # device=0 will use the GPU if available, change to -1 to force CPU.
# manipulation_classifier = pipeline(
#     "text-classification",
#     model=final_model_path,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1
# )
#
# # Test with new example dialogues
# sample_manipulative_dialogue = "After all I've done for you, you're going to say no? I can't believe how ungrateful you are. Fine, do what you want, but don't come crying to me when it all goes wrong."
# sample_normal_dialogue = "I'm heading to the store to pick up some groceries. Do you need anything while I'm out?"
#
# # The model will output 'LABEL_1' (manipulative) or 'LABEL_0' (not manipulative).
# predictions = manipulation_classifier([sample_manipulative_dialogue, sample_normal_dialogue])
#
# print("\n--- Inference Results ---")
# for dialogue, pred in zip([sample_manipulative_dialogue, sample_normal_dialogue], predictions):
#     print(f"\nDialogue: '{dialogue}'")
#     if pred['label'] == 'LABEL_1':
#         print(f"--> Prediction: Manipulative (Score: {pred['score']:.4f})")
#     else:
#         print(f"--> Prediction: Not Manipulative (Score: {pred['score']:.4f})")
#
# print("\n\nScript finished successfully from start to end!")

# ==============================================================================
# Phase 6: Final Evaluation
# ==============================================================================
# The trainer.train() process already ran a final evaluation on our validation set
# because we set load_best_model_at_end=True. The results were printed above.
# There is no separate "test" set in the original data, so this phase is complete.
print("\n--- Starting Phase 6: Final Evaluation ---")
print("The final evaluation results on the validation set are shown above the 'Fine-tuning complete' message.")
print("Since the original dataset does not have a separate 'test' set, this evaluation is considered final.")


# ==============================================================================
# Phase 7: Use the Model for Inference
# ==============================================================================
print("\n--- Starting Phase 7: Inference with New Text ---")
# The Trainer already saved the best model to a checkpoint directory.
# We will save it again to a clean 'final' directory for easy access.
final_model_path = f"./{output_dir}/final"
trainer.save_model(final_model_path)
print(f"Best model saved to {final_model_path} for easy future use.")

# Load the fine-tuned model into a pipeline for easy prediction.
# device=0 will use the GPU if available, change to -1 to force CPU.
manipulation_classifier = pipeline(
    "text-classification",
    model=final_model_path,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

sample_manipulative_dialogue = "After all I've done for you, you're going to say no? I can't believe how ungrateful you are. Fine, do what you want, but don't come crying to me when it all goes wrong."
sample_normal_dialogue = "I'm heading to the store to pick up some groceries. Do you need anything while I'm out?"

predictions = manipulation_classifier([sample_manipulative_dialogue, sample_normal_dialogue])

print("\n--- Inference Results ---")
for dialogue, pred in zip([sample_manipulative_dialogue, sample_normal_dialogue], predictions):
    print(f"\nDialogue: '{dialogue}'")
    if pred['label'] == 'LABEL_1':
        print(f"--> Prediction: Manipulative (Score: {pred['score']:.4f})")
    else:
        print(f"--> Prediction: Not Manipulative (Score: {pred['score']:.4f})")

print("\n\nScript finished successfully from start to end!")