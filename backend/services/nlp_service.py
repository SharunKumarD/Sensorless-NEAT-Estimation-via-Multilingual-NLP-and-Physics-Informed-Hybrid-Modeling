from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

ner_pipeline = None

def init_ner():
    global ner_pipeline
    if ner_pipeline is None:
        try:
            # We initialize XLM-RoBERTa for multilingual NER.
            # To extract 'Action', 'Duration', and 'Environment' PRECISELY, 
            # you would normally point this to your fine-tuned XLM-RoBERTa model on HuggingFace.
            model_name = "xlm-roberta-large-finetuned-conll03-english" # Using a baseline for setup
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception as e:
            print(f"Error loading NER model: {e}")

def extract_entities(text: str):
    init_ner()
    
    # Initialize the required keys
    extracted = {
        "Action": [],
        "Duration": [],
        "Environment": []
    }
    
    if ner_pipeline:
        try:
            results = ner_pipeline(text)
            for entity in results:
                group = entity.get('entity_group', '').upper()
                
                # Assuming fine-tuned labels contain ACTION, DURATION, or LOC/ENV
                if "ACTION" in group:
                    extracted["Action"].append(entity['word'])
                elif "DURATION" in group:
                    extracted["Duration"].append(entity['word'])
                elif "LOC" in group or "ENV" in group:
                    extracted["Environment"].append(entity['word'])
        except Exception as e:
            print(f"Inference error: {e}")
            
    # Return formatted strings from extracted words
    return {
        "Action": " ".join(extracted["Action"]) if extracted["Action"] else None,
        "Duration": " ".join(extracted["Duration"]) if extracted["Duration"] else None,
        "Environment": " ".join(extracted["Environment"]) if extracted["Environment"] else None
    }
