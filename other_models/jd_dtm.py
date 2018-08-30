# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 22:06:45 2018

@author: Leonova

Objective: How well does hierarchical clustering group similar job titles? 
Do senior titles get grouped?
"""

# ===========================
# ========== Set up =========
# ===========================
# Import relevant modules
import os
import pandas as pd

from numpy import count_nonzero
from sklearn.feature_extraction import text 




# File pathways
main_folder = '/Users/Leonova/Repos/jd-classifier/'
jd_folder = '/Users/Leonova/Repos/jd-classifier/jd_files/'
#'/Users/Leonova/Dropbox/8. meDATAtion/Python - Job Mapping/NLP Data School/JDs Training & Test Docx'
image_folder = '/Users/Leonova/Repos/jd-classifier/images/'
model_folder = '/Users/Leonova/Repos/jd-classifier/other_models'


# Change directory
os.chdir(main_folder)
# os.getcwd()

# Import special functions
import clean_data as cld

# ==============================
# ========= Load Data ==========
# ==============================

# The following two variables will be re-used in subsequent cell blocks
titleA = "Data Scientist"
titleB = "Analyst"

# Use the above function to create a data frame of jobs
jds = cld.create_corpus_df(
        roles_folder = jd_folder, 
        titleA = titleA, 
        identifierA = 'Scientist', 
        titleB = titleB,
        titleB_primary = False
        )
        

# Create a more concise dataframe with just two columns
j = pd.DataFrame.copy(jds[['is_primary_role', 'description']])

# Export results into a csv
j.to_csv(os.path.join(model_folder, 'corpus.csv'))

# ======================================
# ======= Document Term Matrix =========
# ======================================
# Create a special list of words that give my model an unfair predictive power
# These are some of the key words I am trying to predict, so I want to see how the model fairs without these extra helpers
my_additional_stop_words = ['scientist', 
                            'scientists',
                            'data scientist', 
                            'data scientists', 
                            'analyst', 
                            'financial',        # Only reason I am removing is because my current sample happens to lean more on the financial analyst roles
                            'data science',
                            'science'           # though we have computer science, we also have data science
                            ] 
# Add my modified list of exceptions to the default list of stopwords                
modified_stop_word_list = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

# Modify the default parameters based on intuition and previous exploration oh other paramters
vect = text.CountVectorizer(
        binary=True,                                 # Count the word once per document, even if it appears multiple times (I want to not give extra emphasis to a word if it appears 10 times in one document, but rarerly in others)  
        ngram_range=(1, 2),                          # Capture 1-2 term long tokens
        #token_pattern=r'(?u)\b\w\w+\b|r|C',         # R (should be included in the vocabulary)
        token_pattern=r'(?u)\b\w+\b',                # Keep single character letters
        stop_words = modified_stop_word_list,        # Including Data Science in the JD is 'cheating'
        min_df=2                                     # The term appears in at least 2 documents (may want to increase this threshold once data sample is larger)
        )


''' A note about parameter "token_pattern":
default token_pattern='(?u)\b\w\w+\b'
(?u):  switches on the re.U (re.UNICODE) flag
\b  :  assert position at a word boundary (similar to carrot or dollar sign) matches a position character that has a word character on one side, and something that's not a word character on the other
\w  :  word character (letter, digit, underscore)
\w+ :  1 or more occurrences of 'w' (the pattern directly to its left)
    
'''
## Other parameters not used:
# max_df=.5 : ignore terms that appear in more than 50% of the documents
# max_df=1.0  : ignore terms that appear in all the documents 
# max_features=1000  : only keep the top 1000 most frequent terms



# Split the data into two vectors
X = j.description
y = j.is_primary_role

# Convert descriptions into a DTM
vect.fit(X)
X_dtm = vect.transform(X)



# Convert the sparse DTM to a dense matrix
a = X_dtm.todense()

# Calculate the sparsity of the matrix
sparsity = 1.0 - count_nonzero(a) / a.size



b = pd.DataFrame(data=a, columns=sorted(vect.vocabulary_))
b.to_csv("dense_dtm.csv")
