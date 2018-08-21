# Job Classification Model

## Motivation
Having completed [Data School: Machine Learning with Text](https://www.dataschool.io/learn/) course, I wanted to apply the concepts I learned to answer a question I frequently got asked: *What is the difference between a data scientist and a data analyst?*

## Summary
I created a supervised learning model on a subset of job postings and used the keywords to predict whether a job description was for a data scientist or data analyst. I then examined the false negatives and false positives and extracted the frequent keywords/phrases to improve the model and learn more about the roles.

## Results
My Multinomial Naive Bayes model was able to predict 73% of job cases correctly, with the default 0.5 threshold.  and had an AUC of 73%, meaning the model did fairly well overall to account for both the true positive rate (sensitivity) and the false positive rate (100-specificity).   

I was happy to see that the top terms that appeared for each role were in line with what I expected. 

For a more detailed write up of the results, please see my blog post ().
 
## Process

#### Data collection: 
I collected a total of 34 unique jobs primarily from big tech companies in Silicon Valley and a handful of smaller companies.

#### Data prepartion: 
I used `CountVectorizer()`, a python scikit-learn library, to create the document-term matrix.I further optimized my model performance by modifying the default parameters such as the n-gram size, the length of tokens, token frequency within each document and the list of stop words.

#### Feature Engineering:

#### Model Selection: 
I tried logistic regression, but found multinomial naive bayes to return better results.

#### Model Validation: 
- I created a pipeline for the tokenizer and model. 
- I used cross-validation to test my model accuracy.

#### Token Exploration:
- Function `create_token_df()`: Creates a dataframe of count of tokens and their weighted frequency for each role. 
- Chart - Tornado: Isolate which terms appear most frequently for a given role. 
- Chart - Venn Diagram: Shows a count of common and unqiue terms for each role, conditional on having those terms appear in at least 30% of jop postings per role.
- Function `extract_surrounding_text()`: Extracts n number of characters that surround a given word/phrase in a text file with the goal of learning more about the context in which a given keyword was used in.


## Next Steps
- Collect more job samples using `BeautifulSoup`
- Further tune the model parameters using `GridSearchCV`
