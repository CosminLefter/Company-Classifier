import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load datasets
company_list = pd.read_csv("ml_insurance_challenge.csv")  # Ensure the file is available
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

# Combine multiple columns into a single text column
company_list["combined_text"] = (company_list["description"] + " " + 
                                 company_list["business_tags"] + " " + 
                                 company_list["sector"] + " " + 
                                 company_list["category"] + " " + 
                                 company_list["niche"]).fillna('')

# Preprocess the combined text column
company_list["cleaned_combined_text"] = company_list["combined_text"].apply(preprocess_text)

# Preprocess taxonomy labels
taxonomy_labels = insurance_taxonomy["label"].apply(preprocess_text)

# Vectorize the combined text
vectorizer = TfidfVectorizer()
X_companies = vectorizer.fit_transform(company_list["cleaned_combined_text"])
X_taxonomy = vectorizer.transform(taxonomy_labels)

# Compute the cosine similarity between company descriptions and taxonomy labels
similarity_matrix = cosine_similarity(X_companies, X_taxonomy)
similarity_df = pd.DataFrame(similarity_matrix, index=company_list["combined_text"], columns=insurance_taxonomy["label"])

# Plot the distribution of cosine similarity scores
plt.figure(figsize=(10, 6))
plt.hist(similarity_matrix.flatten(), bins=50, edgecolor='black')
plt.title('Distribution of Cosine Similarity Scores')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.3  # Set threshold to 0.3
y_labels = (similarity_matrix > threshold).astype(int)  # Assign labels based on similarity exceeding the threshold

# Train the classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_classifier = MultiOutputClassifier(rf_classifier)
multi_classifier.fit(X_companies, y_labels)

# Handle multi-label classification output probabilities individually
predictions_proba = []

# Get probability predictions for each classifier in the MultiOutputClassifier
for clf in multi_classifier.estimators_:
    prob = clf.predict_proba(X_companies)
    predictions_proba.append(prob)

# Now predictions_proba is a list of probabilities for each label
# Flatten it appropriately and threshold at 0.5
predictions = []

for prob in zip(*predictions_proba):  # Iterating over all the label probabilities
    predictions.append([1 if len(p) > 1 and p[1] > 0.5 else 0 for p in prob])  # Check length for binary vs multi-class

predictions = np.array(predictions)

# Assign multiple labels based on predictions with explainability
explanation = []
assigned_labels = []

# Convert the list of probabilities into a list so that it can be indexed
probabilities_list = list(zip(*predictions_proba))

for idx, pred_row in enumerate(predictions):
    assigned = insurance_taxonomy.iloc[np.where(pred_row > 0)]["label"].values
    assigned_labels.append(", ".join(assigned))

    explanation.append({
        label: prob[1] if len(prob) > 1 else prob[0]  # Handling cases where only 1 class is returned
        for label, prob in zip(insurance_taxonomy["label"], probabilities_list[idx])
        if pred_row[list(insurance_taxonomy["label"]).index(label)] > 0
    })

company_list["insurance_label"] = assigned_labels
company_list["explanation"] = explanation

# Save annotated dataset
company_list.to_csv("Tf_IdfVectorization.csv", index=False)
similarity_df.to_csv("Tf_IdfVectorization_Similarity.csv", index=True)
print("completed!")