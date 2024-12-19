import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd

# Step 1: Load Models and Tokenizers for GPT-2 and RAG
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

# Load the original GPT-2 model
original_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the fine-tuned GPT-2 model (assuming it's fine-tuned on QA dataset)
fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")

# RAG Model
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Step 2: Define a Function for Similarity Calculation
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity_score(response, factual_data):
    if factual_data is None:
        return None

    response_embedding = similarity_model.encode([response])
    factual_embedding = similarity_model.encode([factual_data])
    similarity = cosine_similarity(response_embedding, factual_embedding)
    return similarity[0][0]

def is_hallucination(similarity_score, threshold=0.5):
    return similarity_score < threshold

# Step 3: Define a Function to Query the Model
def query_model(question, model, tokenizer, retriever=None):
    if retriever:
        inputs = rag_tokenizer(question, return_tensors="pt")
        retrieved_docs = retriever(question, **inputs)
        generated_ids = rag_model.generate(input_ids=inputs["input_ids"], 
                                           doc_scores=retrieved_docs["doc_scores"], 
                                           doc_input_ids=retrieved_docs["input_ids"])
        response = rag_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(question, return_tensors="pt")
        output = model.generate(input_ids, max_length=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Step 4: Fetch PubMed Data (as Factual Data)
def fetch_pubmed_data(query):
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=5&retmode=xml"
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch PubMed data: {e}")
        return None

# Step 5: Evaluate the Model Output
def evaluate_llm_response(question, factual_data, model, tokenizer, retriever=None):
    response = query_model(question, model, tokenizer, retriever)
    similarity_score = calculate_similarity_score(response, factual_data)
    hallucination_flag = is_hallucination(similarity_score)

    return {
        "question": question,
        "response": response,
        "similarity_score": similarity_score,
        "is_hallucination": hallucination_flag
    }

# Step 6: Test with Question
test_question = "What are the symptoms of diabetes?"
factual_data = fetch_pubmed_data(test_question)

if factual_data is not None:
    # Evaluation for original GPT-2
    original_evaluation = evaluate_llm_response(test_question, factual_data, original_model, tokenizer)
    
    # Evaluation for fine-tuned GPT-2
    fine_tuned_evaluation = evaluate_llm_response(test_question, factual_data, fine_tuned_model, tokenizer)
    
    # Evaluation for RAG
    rag_evaluation = evaluate_llm_response(test_question, factual_data, rag_model, rag_tokenizer, rag_retriever)

    # Output Results
    print("Original GPT-2 Evaluation:")
    print(pd.DataFrame([original_evaluation]))
    
    print("\nFine-Tuned GPT-2 Evaluation:")
    print(pd.DataFrame([fine_tuned_evaluation]))

    print("\nRAG Model Evaluation:")
    print(pd.DataFrame([rag_evaluation]))

else:
    print("Factual data could not be retrieved. Evaluation aborted.")
