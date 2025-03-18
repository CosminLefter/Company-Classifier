To solve this problem I choose cosine similarity metric.

I solved the problem in two ways. First we are using multi-label text clasification with TF-IDF vectorization and Random Forest as the classifier. 
In the second aproach we are using the pre-trained model RoBERTa insteed of the TF-IDF. We are using the second aproach beacuse RoBerta allow us to use the GPU for faster computing 
and better scalability in the future.

I've also created two plots using matlib to see the distribution of the results and I've created a new csv file for each where I've put the the text, label and the metric for better undestanding.

My aproach for the RoBERTa model:
 First I prepeard the text by removing non-alphanumeric characters convert to lowercase and tokenize the text into individual words.

 Next step is to combine the colums for each comany into a single string(sector, description, category etc). 
 First observation and the first question that I asked: 
   In the description of this task it says: 
  "Accept a list of companies with associated data:
    – Company Description
    – Business Tags
    – Sector, Category, Niche Classification" 
  It my be better to exclude some of these categories for example "Sector" from the model beacuse the words that they contain are to general and my affect the acuracy. 
  I've seen in the data words like "Manufactoring" or "Services" that could increase the similarity metric and affect the result.

  Next step I loaded the Roberta model and then converted the text into vector embedings. 
  We are cheking if the GPU is avaliable and then set model to evaluation mode.
  We use mean_pooling to convert the text into dense vector emmbedings. Another aproach would be to use Token Pooling and Max Pooling.
  Disclaimar: I don't fully understand at the moment these pooling strategies. It's a work in progress...
  
  Next step was to process the text in batches of 16 beacause I ran out of memory and the script crashed. Depending on the GPU used we can increment the batches for faster perfomance.

  Next step is to compute the cosine Similarity Matrix and then convert it into a dataframe.

  The best aproach here for now is to show the top 3 most relevant labels. After I find a solution to improve the acuracy further I'm planning to use a trashold to show the results.

 In this picture we can see the distribution. 
  
![Figure_123](https://github.com/user-attachments/assets/5fa9969f-1a8b-430f-aa38-bfaca5f29d27)


For the TF-IDF Vectorization solution I transformed textual data into numerical vectors and apply this on both the associated data and the taxonomy label.



![Figuresklearn](https://github.com/user-attachments/assets/56f085c6-0090-4ccb-888e-184fa63f9a24)


