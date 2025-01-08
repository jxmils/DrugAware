import pandas as pd
import dill
import numpy as np
from collections import defaultdict

import pandas as pd
import numpy as np

def process_medications(med_file):
    """
    Process the medication data from MIMIC-III PRESCRIPTIONS.csv file.
    
    Parameters:
        med_file (str): Path to the PRESCRIPTIONS.csv file.
        
    Returns:
        pd.DataFrame: Processed medication DataFrame.
    """
    # Load medication data
    med_df = pd.read_csv(med_file, dtype={'NDC': 'str'})
    
    # Drop unnecessary columns
    columns_to_drop = [
        'ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
        'FORMULARY_DRUG_CD', 'PROD_STRENGTH', 'DOSE_VAL_RX',
        'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'GSN',
        'ROUTE', 'ENDDATE', 'DRUG'
    ]
    med_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Filter out invalid NDC codes
    med_df = med_df[med_df['NDC'].notnull() & (med_df['NDC'] != '0')]
    
    # Handle missing values
    med_df.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_df['STARTDATE'] = pd.to_datetime(med_df['STARTDATE'], errors='coerce')
    med_df.dropna(subset=['STARTDATE'], inplace=True)
    
    # Forward-fill missing values within each patient stay
    med_df.groupby(['SUBJECT_ID', 'HADM_ID']).fillna(method='ffill', inplace=True)
    
    # Remove duplicates
    med_df.drop_duplicates(inplace=True)
    
    # Reset index
    med_df.reset_index(drop=True, inplace=True)
    
    return med_df

def process_diagnoses(diag_file):
    """
    Process the diagnosis data from MIMIC-III DIAGNOSES_ICD.csv file.

    Parameters:
        diag_file (str): Path to the DIAGNOSES_ICD.csv file.

    Returns:
        pd.DataFrame: Processed diagnosis DataFrame.
    """
    diag_df = pd.read_csv(diag_file, dtype={'ICD9_CODE': 'str'})
    
    # Drop unnecessary columns
    diag_df.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True, errors='ignore')
    
    # Remove rows with missing ICD9_CODE
    diag_df.dropna(subset=['ICD9_CODE'], inplace=True)
    
    # Remove duplicates and reset index
    diag_df.drop_duplicates(inplace=True)
    diag_df.reset_index(drop=True, inplace=True)
    
    return diag_df


def process_procedures(proc_file):
    """
    Process the procedure data from MIMIC-III PROCEDURES_ICD.csv file.

    Parameters:
        proc_file (str): Path to the PROCEDURES_ICD.csv file.

    Returns:
        pd.DataFrame: Processed procedure DataFrame.
    """
    proc_df = pd.read_csv(proc_file, dtype={'ICD9_CODE': 'str'})
    
    # Drop unnecessary columns
    proc_df.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True, errors='ignore')
    
    # Remove rows with missing ICD9_CODE
    proc_df.dropna(subset=['ICD9_CODE'], inplace=True)
    
    # Remove duplicates and reset index
    proc_df.drop_duplicates(inplace=True)
    proc_df.reset_index(drop=True, inplace=True)
    
    return proc_df


def select_top_codes(df, code_column, top_n):
    """
    Select the top N most common codes in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        code_column (str): Column containing the codes.
        top_n (int): Number of top codes to select.

    Returns:
        pd.DataFrame: Filtered DataFrame with top codes.
    """
    top_codes = df[code_column].value_counts().nlargest(top_n).index
    df = df[df[code_column].isin(top_codes)]
    df.reset_index(drop=True, inplace=True)
    return df

def map_ndc_to_atc4(med_df, ndc_rxnorm_file, ndc2atc_file):
    """
    Map NDC codes to ATC Level 4 codes.
    
    Parameters:
        med_df (pd.DataFrame): Medication DataFrame.
        ndc_rxnorm_file (str): Path to NDC to RxNorm mapping file.
        ndc2atc_file (str): Path to RxNorm to ATC mapping file.
        
    Returns:
        pd.DataFrame: Medication DataFrame with ATC codes.
    """
    # Load NDC to RxNorm mapping
    ndc_rxnorm = pd.read_csv(ndc_rxnorm_file)
    ndc_rxnorm.drop_duplicates(subset=['NDC'], inplace=True)
    
    # Merge with medication data
    med_df = med_df.merge(ndc_rxnorm, on='NDC', how='left')
    
    # Drop rows with missing RxCUI
    med_df.dropna(subset=['RXCUI'], inplace=True)
    
    # Load RxNorm to ATC mapping
    rxnorm_atc = pd.read_csv(ndc2atc_file)
    rxnorm_atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    
    # Merge to get ATC codes
    med_df = med_df.merge(rxnorm_atc[['RXCUI', 'ATC']], on='RXCUI', how='left')
    
    # Drop rows with missing ATC codes
    med_df.dropna(subset=['ATC'], inplace=True)
    
    # Use only the first 4 characters of ATC code
    med_df['ATC4'] = med_df['ATC'].str[:4]
    
    # Drop unnecessary columns
    med_df.drop(columns=['NDC', 'RXCUI', 'ATC'], inplace=True)
    
    # Remove duplicates and reset index
    med_df.drop_duplicates(inplace=True)
    med_df.reset_index(drop=True, inplace=True)
    
    return med_df

def filter_patients_with_multiple_visits(med_df, min_visits=2):
    """
    Filter patients with at least a specified number of admissions.
    
    Parameters:
        med_df (pd.DataFrame): Medication DataFrame.
        min_visits (int): Minimum number of visits required.
        
    Returns:
        pd.DataFrame: Filtered medication DataFrame.
    """
    patient_visit_counts = med_df.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    patients_to_keep = patient_visit_counts[patient_visit_counts >= min_visits].index
    med_df = med_df[med_df['SUBJECT_ID'].isin(patients_to_keep)]
    med_df.reset_index(drop=True, inplace=True)
    return med_df

def select_top_medications(med_df, top_n=300):
    """
    Select the top N most common medications.
    
    Parameters:
        med_df (pd.DataFrame): Medication DataFrame.
        top_n (int): Number of top medications to select.
        
    Returns:
        pd.DataFrame: Medication DataFrame with top medications.
    """
    top_meds = med_df['ATC4'].value_counts().nlargest(top_n).index
    med_df = med_df[med_df['ATC4'].isin(top_meds)]
    med_df.reset_index(drop=True, inplace=True)
    return med_df

def combine_data(med_df, diag_df, proc_df):
    """
    Combine medication, diagnosis, and procedure data.
    
    Parameters:
        med_df (pd.DataFrame): Medication DataFrame.
        diag_df (pd.DataFrame): Diagnosis DataFrame.
        proc_df (pd.DataFrame): Procedure DataFrame.
        
    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    # Ensure all DataFrames have the same patients and admissions
    common_keys = set(med_df['SUBJECT_ID'].unique()) & set(diag_df['SUBJECT_ID'].unique()) & set(proc_df['SUBJECT_ID'].unique())
    med_df = med_df[med_df['SUBJECT_ID'].isin(common_keys)]
    diag_df = diag_df[diag_df['SUBJECT_ID'].isin(common_keys)]
    proc_df = proc_df[proc_df['SUBJECT_ID'].isin(common_keys)]
    
    # Merge DataFrames on SUBJECT_ID and HADM_ID
    merged_df = med_df.merge(diag_df, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    merged_df = merged_df.merge(proc_df, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    # Group by patient and admission to collect codes
    merged_df = merged_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg({
        'ATC4': lambda x: list(set(x)),
        'ICD9_CODE_x': lambda x: list(set(x)),  # Diagnoses
        'ICD9_CODE_y': lambda x: list(set(x))   # Procedures
    }).reset_index()
    
    # Rename columns
    merged_df.rename(columns={'ATC4': 'MED_CODE', 'ICD9_CODE_x': 'DIAG_CODE', 'ICD9_CODE_y': 'PROC_CODE'}, inplace=True)
    
    return merged_df

def create_patient_records(data_df, diag_vocab, proc_vocab, med_vocab):
    """
    Create patient records suitable for model input.
    
    Parameters:
        data_df (pd.DataFrame): Combined DataFrame.
        diag_vocab (Vocab): Diagnosis vocabulary.
        proc_vocab (Vocab): Procedure vocabulary.
        med_vocab (Vocab): Medication vocabulary.
        
    Returns:
        list: List of patient records.
    """
    records = []
    for subject_id, group in data_df.groupby('SUBJECT_ID'):
        patient = []
        for _, row in group.iterrows():
            admission = [
                [diag_vocab.word2idx[code] for code in row['DIAG_CODE']],
                [proc_vocab.word2idx[code] for code in row['PROC_CODE']],
                [med_vocab.word2idx[code] for code in row['MED_CODE']]
            ]
            patient.append(admission)
        records.append(patient)
    
    with open('patient_records.pkl', 'wb') as f:
        dill.dump(records, f)
    
    return records


class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

    def add_codes(self, codes):
        for code in codes:
            if code not in self.word2idx:
                self.word2idx[code] = self.size
                self.idx2word[self.size] = code
                self.size += 1

def create_vocabularies(data_df):
    diag_vocab = Vocab()
    proc_vocab = Vocab()
    med_vocab = Vocab()

    for _, row in data_df.iterrows():
        diag_vocab.add_codes(row['DIAG_CODE'])
        proc_vocab.add_codes(row['PROC_CODE'])
        med_vocab.add_codes(row['MED_CODE'])

    vocabs = {'diag_vocab': diag_vocab, 'proc_vocab': proc_vocab, 'med_vocab': med_vocab}
    with open('vocabs.pkl', 'wb') as f:
        dill.dump(vocabs, f)

    return diag_vocab, proc_vocab, med_vocab


if __name__ == '__main__':
    # Configuration
    med_file = '/Users/jasonmiller/Downloads/Research/src/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv'
    diag_file = '/Users/jasonmiller/Downloads/Research/src/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv'
    proc_file = '/Users/jasonmiller/Downloads/Research/src/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv'

    # NDC to xnorm mapping file
    ndc_rxnorm_file = './ndc2rxnorm_mapping.txt'

    ndc2atc_file = './ndc2atc_level4.csv'

    # drug（CID） to ATC code mapping
    cid_atc = './drug-atc.csv'

    # Process Medications
    med_df = process_medications(med_file)
    med_df = map_ndc_to_atc4(med_df, ndc_rxnorm_file, ndc2atc_file)
    med_df = filter_patients_with_multiple_visits(med_df, min_visits=2)
    med_df = select_top_medications(med_df, top_n=300)
    print('Completed medication processing.')

    # Process Diagnoses
    diag_df = process_diagnoses(diag_file)
    diag_df = select_top_codes(diag_df, code_column='ICD9_CODE', top_n=2000)
    print('Completed diagnosis processing.')

    # Process Procedures
    proc_df = process_procedures(proc_file)
    proc_df = select_top_codes(proc_df, code_column='ICD9_CODE', top_n=1000)
    print('Completed procedure processing.')

    # Combine Data
    data_df = combine_data(med_df, diag_df, proc_df)
    data_df.to_pickle('combined_data.pkl')
    print('Completed data combining.')

    # Create Vocabularies
    diag_vocab, proc_vocab, med_vocab = create_vocabularies(data_df)
    dill.dump(
        {'diag_vocab': diag_vocab, 'proc_vocab': proc_vocab, 'med_vocab': med_vocab},
        open('vocabularies.pkl', 'wb')
    )
    print('Created vocabularies.')

    # Create Patient Records
    records = create_patient_records(data_df, diag_vocab, proc_vocab, med_vocab)
    dill.dump(records, open('patient_records.pkl', 'wb'))
    print('Created patient records.')

    print('Pipeline execution completed successfully!')