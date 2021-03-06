#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:13:09 2018
Updated on Thurs Aug 16 11:03:00 2018

@author: paula
"""


# ===================================================
# ====================== Set Up =====================
# ===================================================
"""
Import relevant modules all at once
I have also added a comment in each code section for relevant modules
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib_venn import venn2
from pylab import *


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics


import glob
import docx

import re
import itertools

from sklearn.feature_extraction import text 
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


# Folder pathways
github_function_folder = '/Users/Leonova/Repos/jd-classifier/'
jd_folder = '/Users/Leonova/Dropbox/8. meDATAtion/Python - Job Mapping/NLP Data School/JDs Training & Test Docx'
github_image_folder = '/Users/Leonova/Repos/jd-classifier/images/'

# Import special functions
os.chdir(github_function_folder)
import clean_data as cld


# =====================================================
# ==================== Data Prep ======================
# =====================================================

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


# Split the data into two vectors
X = j.description
y = j.is_primary_role

# ============================================================
# ============= Feature Exploration/Engineering  =============
# ============================================================
#import re
#import itertools 
#  Useful for testing regular expressions: https://regex101.com/#python

# ----------------------------------------
# --------- Years of Experience ----------
# ----------------------------------------
def years_of_experience(df, column_name):         
    min_years_experience = []
    experience_range_years = []  
    for d in range(len(df)):
        string = df[column_name][d]
        match = re.search(r'(\d*)-*[ to ]*(\d)\+* years?[ of ]*.* experience', string)      
        if match:
            # Match all rather than just the first one like you do in re.search
            list_with_potential_tuples = re.findall(r'(\d*)-*[ to ]*(\d)\+* years?[ of ]*.* experience', string)
            simple_list = list(itertools.chain(* list_with_potential_tuples))
            # Remove empty matches
            simple_list = [x for x in simple_list if x != '']
            min_years_experience.append(min(simple_list))
            experience_range_years.append((pd.to_numeric(max(simple_list))-
                                           pd.to_numeric(min(simple_list))) )
        else:
            min_years_experience.append(None)
            experience_range_years.append(None)
    df['min_years'] = pd.to_numeric(min_years_experience)
    df['experience_year_range'] = pd.to_numeric(experience_range_years)
    #return df

years_of_experience(jds, 'description')

# ----------------------------------------------
# ------------ Contains Keywords -----------
# ----------------------------------------------
re_senior_titles = re.compile(r'.* (principal|sr\.|sr|senior).*', flags = re.IGNORECASE)

def is_a_match(df, column_name, regular_expression, new_column_name):
    is_true_column = [] 
    for d in range(len(df)):
        string = df[column_name][d]
        extract_match_ex = re.findall(regular_expression, string)
        contains_match_ex = (extract_match_ex != [])*1
        is_true_column.append(contains_match_ex)
    df[new_column_name] = is_true_column
    #return df

# make sure that the index doesn't skip any (reset_index())
# Create a column for that flags all senior roles (TARGET VARIABLE)
is_a_match(jds, 'title', re_senior_titles, 'is_senior_title')



# Create a function that adds new features as columns to the dataframe
def make_features(df):
    # Create two columns (1) minimum years of experience (2) range from min to max years of experience
    years_of_experience(df, 'description')



# ----------------------------------------------
# ------------ CHART: Boxplot & Violin ---------
# ----------------------------------------------

#import seaborn as sns   
'''
Is there a difference between the years of experience for the roles?
Is the distribution still skewed after removing senior titles?
Is there a difference between the years of experience for senior roles? (will need more data)
'''
d = jds[jds['min_years'].notnull()]
x_val = 'is_primary_role'
y_col=  'min_years'
color_set = ['sandybrown','cornflowerblue']


# Graph 1
sns.boxplot(x='is_primary_role', y='min_years', data=d, palette=color_set)

# Graph 2a
sns.set(font_scale = 1)
ax = sns.boxplot(x='is_senior_title', y='min_years', hue='is_primary_role', data=d, palette=color_set)
# Graph 2b
ax = sns.violinplot(x='is_senior_title', y='min_years', hue='is_primary_role', data=d, palette=color_set, cut = 0)



# Graph 3
# Remove senior roles
jds[(jds['min_years'].notnull()) & jds['is_senior_title']==0]
plt.clf()
sns.set(font_scale = 2)
plt.figure(figsize=(8, 6))
#ax = sns.violinplot(x=x_val, y=y_col, data=d, palette=color_set, cut=0)
ax = sns.boxplot(x=x_val, y=y_col, data=d, palette=color_set)
ax = sns.swarmplot(x=x_val, y=y_col, data=d, color='grey')

medians = d.groupby([x_val])[y_col].median().values
median_labels = [str(np.round(s, 2)) for s in medians]
pos = range(len(medians))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.10, median_labels[tick], 
            horizontalalignment='center', size='x-small', color='w', weight='semibold')

x_val = 'short_title'






# ============================================================
# =========== Data Preprocessing & Model Optimization ========
# ============================================================

# -------------------------------------
# ------------- Tokenizer -------------
# -------------------------------------
#from sklearn.feature_extraction import text 

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
vect = CountVectorizer(binary=True,                                 # Count the word once per document, even if it appears multiple times (I want to not give extra emphasis to a word if it appears 10 times in one document, but rarerly in others)  
                       ngram_range=(1, 2),                          # Capture 1 to 3 length word combinations (there might be some important phrases)
                       #token_pattern=r'(?u)\b\w\w+\b|r|C',          # R (should be included in the vocabulary)
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


# -------------------------------------
# ------------- Multi NB--- -----------
# -------------------------------------
#from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()





# ============================================================
# ==================== Model Validation  =====================
# ============================================================

# -------------------------------------
# -------------- Pipeline -------------
# -------------------------------------
# By creating a pipeline, you are ensuring that you use cross-validataion correctly
# create a pipeline of vectorization and Naive Bayes
# from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vect, nb)

# examine the pipeline steps
pipe.steps

# -------------------------------------
# ---------- Cross Validation ---------
# -------------------------------------
# Cross-validate the entire pipeline 
# from sklearn.cross_validation import cross_val_score
cross_val_score(pipe, X, y, cv=10, scoring='accuracy').mean()
cross_val_score(pipe, X, y, cv=10, scoring='roc_auc').mean()





# ----------------------------------------------
# ------------ Train and Test ---------
# ----------------------------------------------
#from sklearn.cross_validation import train_test_split

# (1) First split into train and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
# (2*) Then split the remaining train into train and validation 
# *use this if you have a lot of data and don't want contanimate the test sample as you train your model
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3)

# Examine Shape of Data and save the X_val for later analysis
print(X_train.shape)
print(X_test.shape)
#print(X_val.shape)  

# How many features are in our training/test set
print(y_train.value_counts())
print(y_test.value_counts())


# -------------------------------------
# ------ Fit Model on Train Set--------
# -------------------------------------
# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
# Examine the fitted tokens
# vect.get_feature_names()

X_test_dtm = vect.transform(X_test)

nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
y_pred_prob = nb.predict_proba(X_test_dtm)


# Explore the accuracy of this single fold validation
#from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# -------------------------------------
# ---------- String Class Fit -----------
# -------------------------------------

# Load in completely new data
jd_folder2 = '/Users/Leonova/Dropbox/8. meDATAtion/Python - Job Mapping/NLP Data School/JDs Training & Test Docx - 2'
jds_plus = cld.create_corpus_df(
        roles_folder = jd_folder2, 
        titleA = titleA, 
        identifierA = 'Scientist', 
        titleB = titleB,
        titleB_primary = False
        )

## How the model would classify specific strings?
# Select the description of just one Job Description
x_title = ""
a = jds_plus.loc[jds_plus['title']==x_title, ['description']].values

## Select a string to Explore 
a = 'machine learning is so fly, reporting is important too'
a = 'Experience implementing machine learning models using TensorFlow, scikit-learn, or other libraries'
a = 'Experience analyzing and reporting on product, user, and/or business data at a tech startup'
a = 'Experience analyzing large, disparate data sets and identifying trends, patterns, and anomalies Ability to translate business questions into concrete data analyses Ability to interpret and present the results of analyses in the context of their original questions Knowledge of statistics and experience building statistical models Experience implementing machine learning models using TensorFlow, scikit-learn, or other libraries Proficiency in R, Python, or other data analysis tools Fluency in SQL'
a = 'Experience analyzing large, disparate data sets and identifying trends, patterns, and anomalies Ability to translate business questions into concrete data analyses Ability to interpret and present the results of analyses in the context of their original questions Knowledge of statistics and experience building statistical models Experience implementing machine learning models using TensorFlow, scikit-learn, or other libraries Proficiency in R, Python, or other data analysis tools Fluency in SQL Preferred Experience analyzing and reporting on product, user, and/or business data at a tech startup Experience evaluating investment opportunities, for example at a bank, VC firm, or hedge fund'
a = 'Experience analyzing and reporting on product, user, and/or business data at a tech startup Experience evaluating investment opportunities, for example at a bank, VC firm, or hedge fund'
a = 'product'
a = 'We are looking for data scientists who love exploring data and uncovering insights. Our team is creating innovative data-driven solutions that are impacting critical business decisions for Goodwater and our portfolio companies. Do these qualities describe you? Curiosity: You notice patterns and phenomena and ask “why?” You’re hungry to learn the broader context and consequences of your work. Quantitative and qualitative thinking: You seek out the best available data and explore it to answer “why?” Your analyses and insights are guided and vetted by your qualitative understanding of consumer behavior and market trends. Humble and collaborative: You have come to realize that the more knowledge you gain, the less you actually know, and that pushes you to continually seek knowledge from others. You are also eager to share your knowledge and insights with others. Entrepreneurial: You’re comfortable with moving fast and adapting to fluid roadmaps and priorities. You aren’t afraid of failure and look for creative solutions to overcome challenges. If this sounds like you, here are some qualifications we’re looking for:'

# Explore results for additional data
string = vect.transform([str(a)])
nb.predict(string)
nb.predict_proba(string)



# -------------------------------------
# ---------- Confusion Matrix -----------
# -------------------------------------

# print the confusion matrix (top right: false positives, bottom left: false negatives)
metrics.confusion_matrix(y_test, y_pred_class)

#  --------------- False Positives/False Negatives ---------------- #
# print message text for the false positives (meaning they were incorrectly classified as spam)
# example: A man being told he is pregnant..
falsePositives = X_test[y_test < y_pred_class]
print(falsePositives)
# Specifically examine the false_positives and the false_negatives in original data
fP = falsePositives.index.values
jds.loc[fP, :]


# print message text for the false negatives (meaning they were incorrectly classified as ham)
# example: a pregnant woman is told she is not pregnant
falseNegatives = X_test[y_test > y_pred_class]
print(falseNegatives)
fN = falseNegatives.index.values
jds.loc[fN, :]


### Examine original data set but use the index to extract the relevant data (based on whether using train or test)
index_list = X_test.index.values
jds.loc[index_list, :]




# ============================================================
# ===================== Token Exploration ====================
# ============================================================
# Naives bayes essentially used the weighted frequency of each word/phrase
# Isolate the top words/phrases that are associated with each role

def create_token_df(description_vector, classifier_vector, primary_title, secondary_title):
    """ Creates a dataframe that has the counts of documents per token. These
    counts are then weighted based on frequency per class to isolate terms/
    phrases that are most popular, but also, unique for a given class.
    
    PARAMETERS
    description_vector: a series containing the description text

    classifier_vector: a series containing the short jd titles   
    
    primary_title: the short title of the primary job title
    
    secondary_title: the short title of the secondary job title
    """
     # Fit vocab on selected data set
    data_dtm = vect.fit_transform(description_vector)
    nb.fit(data_dtm, classifier_vector)
    # Store the vocabulary of selected data set
    data_tokens = vect.get_feature_names()
   

    # Number of times each token appears across all feature flag = 1 (in this case Data Scientist)
    primary_token_count = nb.feature_count_[1, :]
    # Number of times each token appears across all feature flag = 0 (Data Analyst)
    secondary_token_count = nb.feature_count_[0, :]
    
    # Create a DataFrame of tokens with their separate counts
    t1 = 'primary: ' + primary_title
    t2 = 'secondary: ' + secondary_title
    tokens = pd.DataFrame({'token': data_tokens, 
                           t1: primary_token_count, 
                           t2: secondary_token_count}).set_index('token')
    # Extract the index column into its own column
    tokens = tokens.reset_index()
    
    # Calculate the weighted token frequency per class
    tokens['primary_wf'] = tokens[t1]  / nb.class_count_[1]
    tokens['secondary_wf'] = tokens[t2] / nb.class_count_[0]
    
    # Calculate the difference between the two frequencies
    # to identify the tokens with the greatest divergence between the two roles
    tokens['wf_divergence'] = tokens['primary_wf'] - tokens['secondary_wf']
    tokens['wf_average'] = (tokens['primary_wf'] + tokens['secondary_wf'])/2

    return tokens


# Use entire dataset to create the token dataframe from
token_df = create_token_df(X, y, titleA, titleB)



# -------------------------------------
# ---------- CHART: Tornado -----------
# -------------------------------------
#from pylab import *
import matplotlib.style
plt.style.use('seaborn')

# Set up parameters
primary_title = titleA
secondary_title = titleB
x_axis_max = 1.0
both_titles = "Both Roles"


#### ------ Clean up the token list  ---- ###
## Remove very frequent terms in both jds
#token_df_clean = token_df[token_df['wf_average'] <= 0.85]
token_df_clean = token_df.copy()

## Remove terms that aren't insightful 
term_cleanup = ['s', 'work', 'working',
                'years', 'experience', 'data', 'team',
                'key', 'self', 'using', # Terms to omit for Analyst chart
                'looking' # Terms to omit for Data Scientist chart
                ]

for term in term_cleanup:
    token_df_clean = token_df_clean.where(token_df_clean.values != term).dropna()



# ------------   Specify Chart Sorting and Labels ------------ ####
# Choose which title to sort for
sort_for_title = secondary_title #primary_title #both_titles #primary_title, secondary_title

## Specific to "Both Titles" sort chart
#title_string = "Most Frequent Terms\n for {}".format(sort_for_title)
#subtitle_string = "Note: Sorted by the average appearance frequency rate for the two roles."

## Specific to Single Title sort chart
title_string = "Most Common Distinct Terms \n for {}".format(sort_for_title)
subtitle_string = "Note: The terms are sorted by\nthe difference between the frequency rates."




# ------------  Create Tornado Chart ---------------------
# Reduce dataframe to top features only
if sort_for_title == both_titles:   ## Sort for popularity for BOTH
    topT_df = token_df_clean.sort_values('wf_average', ascending = False).head(20)
    # Re-order results and reset index (top result to appear on the bottom)
    topT_df = topT_df.sort_values('wf_average', ascending = True).reset_index()
    x_axis_max = 1.1
elif sort_for_title == primary_title: 
    topT_df = token_df_clean.sort_values('wf_divergence', ascending = True).tail(20).reset_index()
    outline_chart_color_emphasis = 'cornflowerblue'
else:
    topT_df = token_df_clean.sort_values('wf_divergence', ascending = True).head(20)
    # Re-order results and reset index
    topT_df = topT_df.sort_values('wf_divergence', ascending = False).reset_index()
    outline_chart_color_emphasis = 'sandybrown'


# Set up axis variables
bars = topT_df['token'].tolist()
pos = arange(len(bars))    # the bar centers on the y axis


plt.figure(figsize=(10,8))
# Add grid
#plt.grid(zorder=0, alpha =0.5)

plt.barh(topT_df.index.values, -topT_df['primary_wf'], color='cornflowerblue' , label = primary_title)
plt.barh(topT_df.index.values, topT_df['secondary_wf'], color='sandybrown', label = secondary_title )

# Data Scientist or Analyst Diff
plt.barh(topT_df.index.values, -topT_df['wf_divergence'], color=outline_chart_color_emphasis, edgecolor="black", lw=1) 

plt.yticks(pos, bars, fontsize = 16)
plt.xlim(-x_axis_max,x_axis_max)
plt.xticks(fontsize = 16)
plt.xlabel("Frequency Rate in JD's", fontsize = 16)
plt.ylabel("Terms", fontsize = 16)
# Change tick labels to be positive
locs, labels = plt.xticks()
labels = [abs(item) for item in locs]
plt.xticks(locs, labels)

ttl = plt.title(title_string,fontsize=22, fontweight='bold' )
ttl.set_position([.5, 1.1])

plt.suptitle(subtitle_string, y=-.01, fontsize=14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=14)



plt.savefig(github_image_folder + 'TorandoChart_TermSensitivity_' + sort_for_title.replace(" ", "") + '.png',
            bbox_inches="tight")

# -------------------------------------
# ------- CHART: Bar Graph   ---------
# -------------------------------------
# Term dispersion
column_sort = 'secondary_wf'
plt.title(('Term Dispersion: ' + column_sort),fontsize=22, fontweight='bold' )
plt.bar(token_df.sort_values(column_sort, ascending = False).head(100).reset_index().index.values, 
        token_df.sort_values(column_sort, ascending = False)[column_sort].head(100))
plt.savefig(github_image_folder + ('BarGraph_TermDispersion_' + column_sort) + '.png',
            bbox_inches="tight")
# -------------------------------------
# ------- CHART: Venn Diagram ---------
# -------------------------------------
#from matplotlib_venn import venn2

# Count of terms that appear frequently arcross documents
freq = 0.3
overlap = len(token_df[(token_df['primary_wf'] >= freq) & (token_df['secondary_wf'] >= freq)])
prim = len(token_df[token_df['primary_wf'] >= freq])
sec = len(token_df[token_df['secondary_wf'] >= freq])

# Which terms are frequent but don't appear in the other role?
token_df_freq = token_df[(token_df['primary_wf'] >= freq) & (token_df['secondary_wf'] <= freq)]


# Plot Venn Diagram for frequent terms
plt.figure(figsize=(8,8))

title_string = "Frequent Term Overlap\n per Role"
subtitle_string = ("NOTE: Each term appears in at least {0:.0f}%\n of documents per role.").format(freq*100)
plt.title(title_string,fontsize=22, fontweight='bold' )
plt.suptitle(subtitle_string, y= .15, fontsize=14)

v = venn2(subsets = (prim, sec, overlap), set_labels = (titleA, titleB))
v.get_patch_by_id('A').set_color('cornflowerblue')
v.get_patch_by_id('A').set_alpha(1.0)
v.get_patch_by_id('A').set_edgecolor('black')
v.get_patch_by_id('B').set_color('sandybrown')
v.get_patch_by_id('B').set_alpha(1.0)
v.get_patch_by_id('B').set_edgecolor('black')
v.get_patch_by_id('C').set_color('grey')

for text in v.set_labels:
    text.set_fontsize(16)

for text in v.subset_labels:
    text.set_fontsize(16)
    
plt.savefig(github_image_folder + 'VennDiagram_FrequentTerms.png')


# -------------------------------------
# ---------- Token Context -----------
# -------------------------------------
def extract_surrounding_text(word, context_length, df):
    """ Create a dataframe with the jd title and the surrounding characters 
    for the given term/phrase. This allows the user to explore the key term
    in context of the job description.
    
    PARAMETERS
    word: term or phrase that will be searched for
    
    context_length: how many characters before and after to include around the
    'word'
    
    df: dataframe that contains a 'title' and 'description' column
    """
    # Reset variables
    extracted_text = []
    jd_title = []
    jd_short_title = []
    context_df = pd.DataFrame()

    for row in range(len(jds)):
        string_text = df['description'][row].lower()
        jd_name = df['title'][row]
        jd_short_name = df['short_title'][row]
        # Find the location of all the words in the string   
        loc = [m.start() for m in re.finditer(word, string_text)]
        
        # Loop through the list of locations to find the surrounding characters
        for i in range(len(loc)):
            contextual_string = string_text[loc[i]-context_length:loc[i]+context_length]
            extracted_text.append(contextual_string)
            jd_title.append(jd_name)
            jd_short_title.append(jd_short_name)

    context_df['title'] = jd_title
    context_df['short_title'] = jd_short_title
    context_df['string_context'] = extracted_text
    return context_df



# Find all the job postings that have the following word & surrounding characters
con_df = extract_surrounding_text('python', 30, jds)
# Examine the output
con_df.head(10)



