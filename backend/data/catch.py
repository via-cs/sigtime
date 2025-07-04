import pandas as pd

X_train = pd.read_csv('./ECG5000_New/X_train.csv')
label = pd.read_csv('./ECG5000_New/label.csv')
shapelet_transform = pd.read_csv('./ECG5000_New/shapelet_transform.csv')
match_end = pd.read_csv('./ECG5000_New/match_end.csv')
match_start = pd.read_csv('./ECG5000_New/match_start.csv')
script_all = pd.read_csv('./ECG5000_New/script_all.csv')

X_train_subset = X_train.head(1000)
label_subset = label.head(1000)
shapelet_transform_subset = shapelet_transform.head(1000)
match_end_subset = match_end.head(1000)
match_start_subset = match_start.head(1000)
script_all_subset = script_all.head(1000)

X_train_subset.to_csv('./ECG5000_demo/X_train.csv', index=False)
label_subset.to_csv('./ECG5000_demo/label.csv', index=False)
shapelet_transform_subset.to_csv('./ECG5000_demo/shapelet_transform.csv', index=False)
match_end_subset.to_csv('./ECG5000_demo/match_end.csv', index=False)
match_start_subset.to_csv('./ECG5000_demo/match_start.csv', index=False)
script_all_subset.to_csv('./ECG5000_demo/script_all.csv', index=False)
