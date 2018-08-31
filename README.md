# Job Classification Model

## Motivation
Having completed [Data School: Machine Learning with Text](https://www.dataschool.io/learn/) course, I wanted to apply the concepts I learned to answer a question I frequently got asked: *What is the difference between a Data Scientist and a Data Analyst?*

## Summary
I created a supervised learning model on a subset of job postings and used the keywords to predict whether a job description was for a Data Scientist or Data Analyst. I then examined the false negatives and false positives and extracted the frequent keywords/phrases to improve the model and learn more about the roles.

## Results
My Multinomial Naive Bayes model was able to **predict 76%** of job cases correctly, with the default 0.5 threshold and had an **AUC of 88%**, meaning the model did fairly well overall to account for both the true positive rate (sensitivity) and the false positive rate (100-specificity).   

I was happy to see that the top terms that appeared for each role were in line with what I expected.

<img src="https://github.com/pleonova/jd-classifier/blob/master/images/TorandoChart_TermSensitivity_DataScientist.png" width="400"> <img src="https://github.com/pleonova/jd-classifier/blob/master/images/TorandoChart_TermSensitivity_Analyst.png" width="400" align="right">

For a more detailed write up of the results, please see **[my blog post](https://pleonova.github.io/jd-classification/)**.
 
## Contents

- jd_classification.py
    - This is the main file where the model is built and tested as well as where the token exploration happens (WIP: modularize this more, see Next Steps).
- clean_data.py
    - Will eventually contain all the functions that are used to clean and extract data from the original input files.
- requirements.txt
    - List of all the packages/libraries used in the jd_classification.py
- images (folder)
    - Contains all the chart outputs
- jd_files (folder)
    - Contains all the job descriptions text files
 
## Process

#### Data Collection: 
- I collected a total of 34 unique jobs primarily from big tech companies in Silicon Valley and a handful of smaller companies.

#### Data Preparation:
- I stored all the job descriptions in one folder in the .docx format. Each file's name has the company name followed by the title of the role.
- Function `create_corpus_df()`: Loops through all files in the folder containing the job postings in order to create a dataframe with columns for the the full file name, description, simplified title and yes/no classifier for primary role.

#### Feature Exploration/Engineering:
- Function `years_of_experience()`: Extracts the number of years listed in the job requirements using regular expressions and creates an additional column in the JD dataframe. If a job has a range for years of experience, the difference between the lower and upper bound is added to a separate column.
- Function `is_a_match()`: Finds matches for a given regex for the selected column. (example: creates a yes/no column for whether principal, senior or sr appears in the JD)
- Chart Box-plot and Violin: Explore the distribution of years of experience within the sample of JDs.

#### Data Preprocessing & Model Optimization: 
- I used `CountVectorizer()`, a python scikit-learn library, to create the document-term matrix. I further optimized my model performance by modifying the default parameters such as the n-gram size, the length of tokens, token frequency within each document and the list of stop words.
- I tried logistic regression, but found multinomial naive bayes to return better results.

#### Model Validation:
- I created a pipeline for the tokenizer and model and used cross-validation to test my model accuracy and ROC.
- Split the data into train/test
- Create a confusion matrix and examine the false positives and false negatives

#### Token Exploration:
- Function `create_token_df()`: Creates a dataframe of count of tokens and their weighted frequency for each role. 
- Chart - Tornado: Isolate which terms appear most frequently for a given role. 
- Chart - Venn Diagram: Shows a count of common and unqiue terms for each role, conditional on having those terms appear in at least 30% of jop postings per role.
- Function `extract_surrounding_text()`: Extracts n number of characters that surround a given word/phrase in a text file with the goal of learning more about the context in which a given keyword was used in.

## Next Steps
- [x] Add confusion matrix
- [ ] Split out the main file into smaller code files for easier management
- [ ] Use Latent Dirichlet Allocation to find topic themes
- [ ] Identify which section requirements are listed in (nice to haves, bottom, top, requirements, etc.) 
- [ ] Collect more job samples using `BeautifulSoup`
- [ ] Identify whether there is some sort of regional pattern
- [ ] Further tune the model parameters using `GridSearchCV`
- [ ] Create four separate classifiers (junior and senior for each role)
- [ ] Try a clustering algorithm without a training set to see which jobs get automatically grouped
- [ ] Generalize this model for other job pairings

## Future Questions
- How does my model stand up against future job postings? What new terms will become common? Will the distinction between Data Analyst and Data Scientist become more grey?

