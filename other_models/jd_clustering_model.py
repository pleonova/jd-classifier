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



# File pathways
github_function_folder = '/Users/Leonova/Repos/jd-classifier/'
jd_folder = '/Users/Leonova/Dropbox/8. meDATAtion/Python - Job Mapping/NLP Data School/JDs Training & Test Docx'
github_image_folder = '/Users/Leonova/Repos/jd-classifier/Images/'

# Change directory
os.chdir(github_function_folder)
# os.getcwd()

# Import special functions
import clean_data as cld

        




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



