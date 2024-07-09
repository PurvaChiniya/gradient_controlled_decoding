from datasets import load_dataset, load_from_disk
import json
def preprocess_data(example):
    """
    Preprocess the data by extracting the prompt and response from the JSON data.
    If the data is not in the expected format, return None to skip the example.
    """
    try:
        # Parse the JSON string
        data = json.loads(example['text'])
        
        # Extract the prompt (utterance)
        prompt = data.get('turns', [{}])[0].get('utterance', '')
        
        # Extract the response
        response = data.get('turns', [{}])[0].get('action_plans', [{}])[0].get('actions', [{}])[0].get('response', '')
        
        # Format the data as '<s>[INST] {prompt} [/INST] {response}'
        formatted_data = f"<s>[INST] {prompt} [/INST] {response}"
        
        return {'text':formatted_data}
    
    except Exception as e:
        # If there is any error, return None to skip the example
        print(f"Error processing example: {e}")
        return None
    
    
dataset = load_from_disk("13/data/split.train")

# Apply the preprocessing function
preprocessed_dataset = dataset.map(preprocess_data)
