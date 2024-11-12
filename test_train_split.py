import os
from sklearn.model_selection import train_test_split
import pandas as pd

split_path = 'dataset/youtube_spam/train_dev_test_split'
os.makedirs(split_path, exist_ok=True)

data = pd.read_csv('dataset/youtube_spam/preprocessed/Processed_Youtube_Spam_Dataset.csv')

train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_file_path = os.path.join(split_path, 'train.csv')
val_file_path = os.path.join(split_path, 'dev.csv')
test_file_path = os.path.join(split_path, 'test.csv')

train_data.to_csv(train_file_path, index=False)
val_data.to_csv(val_file_path, index=False)
test_data.to_csv(test_file_path, index=False)
