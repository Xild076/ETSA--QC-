import xml.etree.ElementTree as ET
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging as hf_logging
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

# --- Configuration ---
# Suppress verbose logging from Hugging Face transformers
hf_logging.set_verbosity_error()

# VADER polarity thresholds
VADER_POS_THRESHOLD = 0.1
VADER_NEG_THRESHOLD = -0.1

# File paths
LAPTOP_FILE = 'test_laptop_2014.xml'
RESTAURANT_FILE = 'test_restaurant_2014.xml'

# Define the order of labels for consistent reporting
LABELS = ['positive', 'negative', 'neutral']

# --- Data Parsing ---

def parse_xml_data(file_path):
    """
    Parses the SemEval XML format to extract sentences, aspect terms, and polarities.
    Skips any samples that are not labeled as 'positive', 'negative', or 'neutral'.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        for sentence in root.findall('sentence'):
            text = sentence.find('text').text
            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is not None:
                for aspect_term in aspect_terms.findall('aspectTerm'):
                    polarity = aspect_term.get('polarity')
                    if polarity in LABELS:
                        data.append({
                            'sentence': text,
                            'aspect_term': aspect_term.get('term'),
                            'polarity': polarity
                        })
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []

# --- Metrics Calculation ---

def calculate_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=LABELS)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=LABELS)
    
    # Generate the detailed report
    report = classification_report(
        y_true, 
        y_pred, 
        labels=LABELS, 
        digits=4,
        zero_division=0
    )
    
    return {
        "accuracy": accuracy * 100,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "classification_report": report
    }

# --- Benchmark Functions ---

def run_vader_benchmark(data):
    """
    Runs VADER sentiment analysis and calculates a full suite of metrics.
    """
    if not data:
        return {}

    sid = SentimentIntensityAnalyzer()
    y_true = []
    y_pred = []
    
    print("Running VADER analysis...")
    for item in tqdm(data):
        sentence = item['sentence']
        
        y_true.append(item['polarity'])
        
        scores = sid.polarity_scores(sentence)
        compound_score = scores['compound']
        
        if compound_score > VADER_POS_THRESHOLD:
            y_pred.append('positive')
        elif compound_score < VADER_NEG_THRESHOLD:
            y_pred.append('negative')
        else:
            y_pred.append('neutral')
            
    return calculate_metrics(y_true, y_pred)

def run_hf_benchmark(data, classifier):
    """
    Runs the Hugging Face ABSA pipeline and calculates a full suite of metrics.
    """
    if not data:
        return {}

    y_true = []
    y_pred = []
    
    print("Running Hugging Face model analysis...")
    start_time = time.time()
    
    for item in tqdm(data):
        sentence = item['sentence']
        aspect = item['aspect_term']
        
        y_true.append(item['polarity'])
        
        result = classifier(sentence, text_pair=aspect)
        
        # The pipeline directly returns the label with the highest score
        predicted_polarity = result[0]['label'].lower()
        y_pred.append(predicted_polarity)
            
    end_time = time.time()
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics['total_time_seconds'] = end_time - start_time
    metrics['total_samples'] = len(data)
    
    return metrics

# --- Result Printing ---

def print_results(dataset_name, model_name, metrics):
    """Neatly prints the calculated metrics."""
    header = f"--- {model_name} Results on {dataset_name} ---"
    print(f"\n{header}")
    
    if not metrics:
        print("No metrics to display.")
        return
        
    print(f"  -> Accuracy:     {metrics['accuracy']:.2f}%")
    print(f"  -> Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  -> Macro F1:     {metrics['macro_f1']:.4f}")
    
    if 'total_time_seconds' in metrics:
        samples = metrics['total_samples']
        time_taken = metrics['total_time_seconds']
        sps = samples / time_taken if time_taken > 0 else 0
        print(f"  -> Time Taken:   {time_taken:.2f} seconds ({sps:.2f} samples/sec)")
        
    print("\n  Classification Report:")
    indented_report = "\n".join(["    " + line for line in metrics['classification_report'].split("\n")])
    print(indented_report)
    print("-" * len(header))

# --- Main Execution ---

def main():
    print("--- Loading and Parsing Datasets ---")
    laptop_data = parse_xml_data(LAPTOP_FILE)
    restaurant_data = parse_xml_data(RESTAURANT_FILE)
    
    if not laptop_data and not restaurant_data:
        print("Could not find or parse any data files. Exiting.")
        return
        
    combined_data = laptop_data + restaurant_data
    
    print(f"Loaded {len(laptop_data)} valid samples from Laptop dataset.")
    print(f"Loaded {len(restaurant_data)} valid samples from Restaurant dataset.")
    print(f"Total valid samples: {len(combined_data)}\n")

    # --- Part 1: VADER Benchmark ---
    print("="*50)
    print("Part 1: VADER Benchmark (Sentence-Level Analysis)")
    print("="*50)
    
    vader_laptop_metrics = run_vader_benchmark(laptop_data)
    vader_resto_metrics = run_vader_benchmark(restaurant_data)
    vader_combined_metrics = run_vader_benchmark(combined_data)
    
    print_results("Laptop Dataset", "VADER", vader_laptop_metrics)
    print_results("Restaurant Dataset", "VADER", vader_resto_metrics)
    print_results("Combined Dataset", "VADER", vader_combined_metrics)

    # --- Part 2: Hugging Face Model Benchmark ---
    print("\n" + "="*50)
    print("Part 2: Hugging Face DeBERTa ABSA Benchmark")
    print("="*50)
    
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    print(f"Loading model '{model_name}'... (This may take a moment)")
    try:
        classifier = pipeline("text-classification", model=model_name, device=0) # Use device=0 for GPU
    except Exception as e:
        print(f"Failed to load the Hugging Face model. Error: {e}")
        return

    print("Model loaded successfully.\n")

    hf_laptop_metrics = run_hf_benchmark(laptop_data, classifier)
    hf_resto_metrics = run_hf_benchmark(restaurant_data, classifier)
    hf_combined_metrics = run_hf_benchmark(combined_data, classifier)

    print_results("Laptop Dataset", "Hugging Face DeBERTa", hf_laptop_metrics)
    print_results("Restaurant Dataset", "Hugging Face DeBERTa", hf_resto_metrics)
    print_results("Combined Dataset", "Hugging Face DeBERTa", hf_combined_metrics)

if __name__ == "__main__":
    main()