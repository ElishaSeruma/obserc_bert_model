**Fine-Tuned BERT for Mental Manipulation Detection**

This project contains the code and instructions for fine-tuning a BERT model to detect manipulative language in conversations. The model is trained on the MentalManip dataset and can classify a given dialogue as either "Manipulative" or "Not Manipulative".

The final trained model achieves approximately 74% accuracy on this task.

**Dataset**

This project would not be possible without the work of the researchers who created the MentalManip dataset. Full credit goes to the original authors.

**Paper: MentalManip:** A Dataset For Fine-grained Analysis of Mental Manipulation in Conversations (ACL 2024)
**Hugging Face Dataset:** audreyeleven/MentalManipProject Structurebert-pycharm-project/
├── bert-mental-manipulation-detector/  # Output directory for the trained model
│   └── checkpoint-984/               # The final saved model files are here
├── run_finetuning.py                   # Script to train the model from scratch
├── use_model.py                        # Script to load and use the trained model for predictions
└── README.md                           # This file

**Setup and Installation**

This project uses Python and several machine learning libraries. Follow these steps to set up the environment.

**1. Create a Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies.
(Create a virtual environment python3 -m venv .venv)

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

**2. Install Dependencies**

The required packages are listed in requirements.txt.pip install -r requirements.txt
You will need to create a requirements.txt file in your project directory with the following content

:# requirements.txt
torch
transformers
datasets
evaluate
scikit-learn
pandas
optimum[exporters]
onnx
onnx-tf

**How to Use**

There are two main scripts: 
one for training the model and one for using the already trained model.
1. Training the Model from ScratchTo run the full fine-tuning process yourself, execute the run_finetuning.py script.Note: The training process is computationally intensive. On a standard CPU, this may take several hours. The script will automatically save the best model checkpoint inside the bert-mental-manipulation-detector/ directory.python run_finetuning.py

2. Using the Pre-Trained Model for PredictionsOnce the model has been trained and saved, you can use the use_model.py script to make predictions on new text.Important: Before running, make sure to update the model_path variable in use_model.py to point to the correct checkpoint directory (e.g., ./bert-mental-manipulation-detector/checkpoint-984).python use_model.py

The script will load the model and run inference on a few example dialogues, printing the predictions and their confidence scores Model Performance The model was trained for 3 epochs. The final performance, evaluated on a validation set (10% of the original training data), is as follows:Evaluation Loss: 0.6535Accuracy: 0.7431F1 Score (Weighted): 0.7304

