import re
import pandas as pd


def preprocess_text(text):
    # 正则表达式移除表情符号，但保留URL中的/
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def main():
    data_path = 'dataset\youtube_spam\Youtube-Spam-Dataset.csv'
    preprocess_path = 'dataset\youtube_spam\preprocessed\Processed_Youtube_Spam_Dataset.csv'

    data = pd.read_csv(data_path)
    data['text'] = data['CONTENT'].apply(preprocess_text)
    data['label'] = data['CLASS']
    
    data = data.dropna()

    processed_data = data[['text', 'label']]

    data = data.dropna()
    data = data[data['text'] != '']
    data = data[data['text'].notna()]

    processed_data.to_csv(preprocess_path, index=False, header=False)

    return None

if __name__ == '__main__':
    main()
