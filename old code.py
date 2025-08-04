from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
# The 'con' version seems to be the main conversational dataset
#try:
#    dataset = load_dataset("audreyeleven/MentalManip", "mentalmanip_con")
#   #dataset = load_dataset("audreyeleven/MentalManip", "con")
#    print("Successfully loaded the 'con' configuration.")
#except Exception as e:
#   print(f"Could not load the 'con' config: {e}")
#   print("Attempting to load the default configuration.")
#   dataset = load_dataset("audreyeleven/MentalManip")


# The dataset is already split into train, validation, and test sets.
#print(dataset)
# Let's inspect a single example from the training set
#print("\nExample from the training set:")
#print(dataset['train'][0])#

##PREVIOUS CODE