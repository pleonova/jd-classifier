# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:32:54 2018

@author: Leonova
"""

import docx
import glob
import os
import pandas as pd



def create_corpus_df(roles_folder, titleA, identifierA, titleB, titleB_primary = False):
    """ Create a dataframe from a folder containing job posting .docx files.
    
    Parameters
    ----------
    roles_folder : str
        Location of folder where all the job .docx files are stored.
    
    titleA : str
        Short title for a subset of jobs in the folder.
    
    identifierA : str
        Part of the job title that is used to distinguish titleA jobs
        (Ex. 'analyst' in Marketing Data Analyst 
        or 'cientist' in "Staff Data Scientist")
    
    titleB : str
        Short title  for other subset of jobs (this is the tile that will
        be given to the rest of the jobs that were not picked 
        up by the identifier)
    
    titleB_primary : bool, optional 
        Default is False, meaning that titleA is our primary role
        and will be 1 in the returned df
        
    Returns
    -------
    A dataframe where each row has the job title (name of the file), 
    shortened/simplified title, full job description, and boolean classifier.
    
    
    """

    # --------------- PART 1: Aggregate all files in the folder  -------------#
    # Change Directory to where the files are stored
    os.chdir(roles_folder)
    
    # All files have been stored as .docx
    text_filenames = glob.glob('*.docx')
    
    def getText(filename):
        doc = docx.Document(filename)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)    
    
    file_text = []
    for filename in text_filenames:
        file_text.append(getText(filename))

     # --------------- PART 2: Clean up the job titles -----------------------#
    # Convert the lists into a DataFrame
    df = pd.DataFrame({'title':text_filenames, 'description':file_text})
    # Clean up the title column
    df['title'] = df['title'].str.replace(".docx", "")
    df['lower_case_title'] = [element.lower() for element in text_filenames]
    
    # Identify if the job contains the key identifier
    idA = str.lower(identifierA)
    df['is_primary_role'] = df.lower_case_title.str.contains(idA).astype(int)
    
    # Instead of using a loop, use the replace method
    df['short_title'] = df['is_primary_role'].replace(1, titleA)
    # Use the short_title column to replace the remaining 0s
    df['short_title'] = df['short_title'].replace(0, titleB)
    
    # Examine how many of each short_title there is
    df['short_title'].value_counts()
    
    # Should the primary role actually be for the second role?
    # The one that didn't have a unique identifier
    if titleB_primary:
        df['is_primary_role'] = 1 - df['is_primary_role']
    
    return df


