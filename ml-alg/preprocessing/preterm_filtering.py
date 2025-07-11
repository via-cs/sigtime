import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse

def parse_weekday_format(week_day_str):
    if week_day_str:
        matching = re.match(r'(\d+)w(\d+)d', week_day_str)
        if matching:
            weeks = int(matching.group(1))
            days = int(matching.group(2))
            return pd.Timedelta(weeks=weeks, days=days)
        else:
            return pd.NaT
    else:
        return pd.NaT

parser = argparse.ArgumentParser()
parser.add_argument('--filter_mode', default='all', choices=['all', 'last', 'last2', 'no_contract'])
args = parser.parse_args()
version = {
    'all': '1', 
    'last': '2',
    'last2': '3',
    'no_contract': '4'
}
if __name__ == "__main__":
    print('filtering...')
    config = args.__dict__
    print(config)
    model_data = pd.read_csv('./data/processed_clinical_data_duplicate.csv') # sheet matches the strip data and detailed metadata
    ID_label = pd.read_excel('./data/case controls PTB 02072025.xlsx', sheet_name='Sheet4') # label with encounter date and delivery date
    ID_match = pd.read_excel('./data/case controls PTB 02072025.xlsx', sheet_name='Sheet2') # label to match mother MRN to obfuscated MRN
    
    model_data['EncDate_ShiftedDate'] = pd.to_datetime(model_data['EncDate_ShiftedDate'], errors='coerce')
    if args.filter_mode == 'no_contract':
        ID_label = ID_label[ID_label['Label']!='PTC_NoPTB']
    else:
        ID_label = ID_label[ID_label['Label']!='No_contractions']
    ID_label.rename(columns={'Mother_obsfucated_MRN': 'Mother_Obfus_MRN'}, inplace=True)
    ID_label['Encounter_date'] = pd.to_datetime(ID_label['Encounter_date'], errors='coerce')
    ID_label['Delivery_date'] = pd.to_datetime(ID_label['Delivery_date'], errors='coerce')
    
    ID_sheet_merge = ID_label.merge(ID_match, on=['Mother_MRN', ], how="left", suffixes=('_label', '_match')) # merge the label and match sheets
    ID_sheet_merge['Mother_Obfus_MRN'] = ID_sheet_merge['Mother_Obfus_MRN_label'].fillna(ID_sheet_merge['Mother_Obfus_MRN_match']) # fill the obfuscated MRN with the match sheet
    ID_filled = ID_sheet_merge[['Mother_MRN', 'Mother_Obfus_MRN', 'Delivery_date', 'Encounter_date', 
                                'Delivery_GA', 'Label', 'PTB', 'Preterm_contractions_on_toco', 'Admission']] # keep only the relevant columns
    ID_filled.to_csv("./data/ID_filled.csv", index=False) # 
    
    model_data['row_number'] = model_data.index
    ID_filled = ID_filled.copy()  # This ensures it's not a slice
    ID_filled.loc[:, 'Delivery_GA_parse'] = ID_filled['Delivery_GA'].apply(parse_weekday_format)
    ID_filled['Pregnant_date'] = ID_filled['Delivery_date'] - ID_filled['Delivery_GA_parse'] # calculate the pregnant date from delivery date and GA
    
    linked_data = pd.merge(model_data, ID_filled, on=['Mother_Obfus_MRN', ], how="inner")
    filtered_data = linked_data[(linked_data['EncDate_ShiftedDate'] >= linked_data['Pregnant_date']) \
        & (linked_data['EncDate_ShiftedDate'] <= linked_data['Delivery_date'])]
    filtered_data = filtered_data.sort_values(by=['Mother_Obfus_MRN', 'EncDate_ShiftedDate'])
    print(filtered_data)
    
    if config['filter_mode'] == 'all':
        filtered_data_keep = filtered_data
    elif config['filter_mode'] == 'last':
        filtered_data_keep = filtered_data.loc[filtered_data.groupby('Mother_Obfus_MRN')['EncDate_ShiftedDate'].idxmax()]
    elif config['filter_mode'] == 'last2':
        # Keep only the second-to-last record for each mother
        def get_second_to_last(df):
            if len(df) >= 2:
                return df.nlargest(2, 'EncDate_ShiftedDate').nsmallest(1, 'EncDate_ShiftedDate')
            else:
                return df  # Return the last (only) record if only one exists
        filtered_data_keep = filtered_data.groupby('Mother_Obfus_MRN', group_keys=False).apply(get_second_to_last).reset_index(drop=True)
    else:  # fallback, keep all
        filtered_data_keep = filtered_data
        
    print(filtered_data_keep)
    with open("../data/strips_data.json", "r") as f:
        ua_list = json.load(f)
        
    waveform = []
    valid_rows = []
    
    for idx, row_number in enumerate(filtered_data_keep['row_number']):
        signal = ua_list[row_number]
        if any(signal):  # Check if waveform is not all zeros
            waveform.append(signal)
            valid_rows.append(idx)
    
    # Keep only valid rows in linked_data
    filtered_data_keep = filtered_data_keep.iloc[valid_rows].reset_index(drop=True)
    
    # Save cleaned linked_data as CSV
    
    filtered_data_keep.to_csv(f"../data/filtered_clinical_data_v{version[config['filter_mode']]}.csv", index=False)
    
    # Save cleaned waveform data as JSON
    with open(f"../data/filtered_strips_data_v{version[config['filter_mode']]}.json", "w") as f:
        json.dump(waveform, f)

