import pandas as pd
import numpy as np
import re
import nltk
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Download necessary NLTK resources. First we check if the resoursces are downloaded.It helps to save time!
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load datasets
company_list = pd.read_csv("ml_insurance_challenge.csv")  
insurance_taxonomy = pd.read_csv("insurance_taxonomy.csv") 

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]

    return " ".join(filtered_tokens)

# Combine and preprocess text
def combine_columns(row):
    return " ".join(filter(None, [row[col] if isinstance(row[col], str) else "" for col in ["description", "business_tags", "sector", "category", "niche"]]))

company_list["combined_text"] = company_list.apply(combine_columns, axis=1)
company_list["cleaned_combined_text"] = company_list["combined_text"].apply(preprocess_text)
insurance_taxonomy["cleaned_label"] = insurance_taxonomy["label"].apply(preprocess_text)

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function for mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / input_mask_expanded.sum(1)

# Function to compute embeddings
def get_roberta_embedding(texts):
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
    
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    embeddings = mean_pooling(outputs, encoded_inputs['attention_mask'])
    return embeddings.cpu().numpy()

# Batch processing for embeddings
def get_embeddings_in_batches(text_batches):
    embeddings = []
    for batch in text_batches:
        batch_embeddings = get_roberta_embedding(batch)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Compute embeddings for company descriptions
batch_size = 16  
company_batches = [company_list["cleaned_combined_text"].iloc[i:i + batch_size].tolist() for i in range(0, len(company_list), batch_size)]
X_companies = get_embeddings_in_batches(company_batches)

# Compute embeddings for insurance taxonomy
X_taxonomy = get_roberta_embedding(insurance_taxonomy["cleaned_label"].tolist())

# Compute cosine similarity
similarity_matrix = cosine_similarity(X_companies, X_taxonomy)

# We see the max and min similarity! 
print("Max Similarity:", np.max(similarity_matrix))
print("Min Similarity:", np.min(similarity_matrix))

# Convert similarity matrix to DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=company_list["combined_text"], columns=insurance_taxonomy["label"])

# Plot histogram of similarity scores
plt.figure(figsize=(10, 6))
plt.hist(similarity_matrix.flatten(), bins=50, edgecolor='black')
plt.title('Distribution of Cosine Similarity Scores')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()

# Assign **top 3** labels based on similarity
assigned_labels = []

for i in range(len(company_list)):
    top_indices = np.argsort(similarity_matrix[i])[-3:][::-1]  # Get indices of top 3 highest values
    top_labels = insurance_taxonomy.iloc[top_indices]["label"].values  
    assigned_labels.append(", ".join(top_labels))  

# Store assigned labels
company_list["top_3_insurance_labels"] = assigned_labels

# Save results
company_list.to_csv("annotated_company_list.csv", index=False)
similarity_df.to_csv("similarity_scores.csv")

print("completed!")