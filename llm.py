import requests
import torch
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline, logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def fetch_pubmed_data(query):
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=5&retmode=xml"
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch PubMed data: {e}")
        return None

def calculate_similarity_score(response, factual_data, similarity_model):
    if factual_data is None:
        return None

    response_embedding = similarity_model.encode([response])
    factual_embedding = similarity_model.encode([factual_data])
    similarity = cosine_similarity(response_embedding, factual_embedding)
    return similarity[0][0]

def is_hallucination(similarity_score, threshold=0.5):
    return similarity_score < threshold

def evaluate_llm_response(question, model, tokenizer, factual_data, similarity_model, device):
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1 if device == "cpu" else 0)
    response = nlp(question=question, context=factual_data)
    answer = response['answer']

    similarity_score = calculate_similarity_score(answer, factual_data, similarity_model)
    hallucination_flag = is_hallucination(similarity_score)

    return {
        "question": question,
        "response": answer,
        "similarity_score": similarity_score,
        "is_hallucination": hallucination_flag
    }

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to use {device}")

    # Suppress warnings
    logging.set_verbosity_error()

    # Load Models
    qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    fine_tuned_model_path = "./fine_tuned_gpt2"  # Update to your fine-tuned model path if different

    original_qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)
    original_qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name).to(device)

    fine_tuned_tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
    fine_tuned_model = BertForQuestionAnswering.from_pretrained(fine_tuned_model_path).to(device)

    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Test question and evaluation
    test_question = "What are the symptoms of diabetes?"
    factual_data = fetch_pubmed_data(test_question)

    if factual_data is not None:
        # Evaluate using original model
        original_result = evaluate_llm_response(test_question, original_qa_model, original_qa_tokenizer, factual_data, similarity_model, device)

        # Evaluate using fine-tuned model
        fine_tuned_result = evaluate_llm_response(test_question, fine_tuned_model, fine_tuned_tokenizer, factual_data, similarity_model, device)

        # Combine results for comparison
        combined_results = pd.DataFrame([
            {"Model": "Original", **original_result},
            {"Model": "Fine-Tuned", **fine_tuned_result}
        ])

        print("Final Evaluation Results:")
        print(combined_results)
    else:
        print("Factual data could not be retrieved. Evaluation aborted.")
