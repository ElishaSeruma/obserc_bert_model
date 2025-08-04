# use_model.py

from transformers import pipeline
import torch

print("Loading the fine-tuned model for inference...")

# This path points to the final model saved by the training script
model_path = "./bert-mental-manipulation-detector/checkpoint-984"

# Load the model into a pipeline for easy prediction
try:
    manipulation_classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path, # The tokenizer is also saved in the same directory
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model. Make sure the training has completed and the model is saved at '{model_path}'.")
    print(f"Error: {e}")
    exit()


# --- Now you can use the model on any text ---

dialogues_to_test = [
    "You wouldn't understand, it's too complicated for you. Just trust me and do as I say.",
    "If you really cared about this team, you'd stay late and finish this. Don't be selfish.",
    "I'm thinking of grabbing pizza for dinner, would you like some?",
    "Could you please review this report when you have a moment?"
]

print("\n--- Running Predictions ---")
predictions = manipulation_classifier(dialogues_to_test)

for dialogue, pred in zip(dialogues_to_test, predictions):
    print(f"\nDialogue: '{dialogue}'")
    # The label is either 'LABEL_0' (not manipulative) or 'LABEL_1' (manipulative)
    if pred['label'] == 'LABEL_1':
        print(f"--> Prediction: Manipulative (Score: {pred['score']:.4f})")
    else:
        print(f"--> Prediction: Not Manipulative (Score: {pred['score']:.4f})")