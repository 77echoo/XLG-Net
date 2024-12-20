import ast
import requests

from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

url = "XXX"  # API URL
username = "XXX"
password = "XXXX"
auth = HTTPBasicAuth(username, password)

model = "gpt-3.5-turbo-4k"
stream = False

parameters = {
    "model": model,
    "messages": [{"role": "user", "content": ""}],
    "temperature": 0,
    "max_tokens": 10,
    "top_p": 0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stream": stream,
    "stop": None
}

dataset = 'R52'

def load_data(data_file, label_file):
    with open(data_file, 'r') as f:
        texts = f.readlines()

    with open(label_file, 'r') as f:
        labels = f.readlines()

    test_texts = []
    test_labels = []

    for line in labels:
        parts = line.strip().split('\t')
        id, data_split, label = parts[0], parts[1], parts[2]

        if data_split == "test":
            test_texts.append(texts[int(id)])
            test_labels.append(label)

    return test_texts, test_labels

def classify_text(texts):
    predictions = []
    for text in tqdm(texts, desc="In the process of classification", ncols=100):
        if dataset == 'mr':
            prompt = f"Please judge the sentiment of the following film reviews：\n'{text}'\n reply ‘positive’ or ‘negative’"
        elif dataset == 'ohsumed':
            prompt = ''
        else:
            prompt = f"I am working on a text classification task using the R52 dataset, which contains 52 categories such as 'earn' (earnings reports), 'crude' (oil markets), 'grain' (grain and agricultural products), and others. Please classify the following sentence into one of the R52 categories and reply only one word.: '{text}'"

        parameters["messages"][0][
            "content"] = prompt

        try:
            response = requests.post(url, auth=auth, json=parameters, verify=False)
            response_data = response.json()

            if response.status_code == 200:
                prediction = response_data['choices'][0]['message']['content'].strip()
                print('\n prediction:', prediction)
                if dataset == 'mr':
                    if prediction == 'positive' or prediction == 'Positive':
                        prediction = 1
                    else:
                        prediction = 0
                predictions.append(prediction)
            else:
                print(f"Error: {response.status_code}, {response_data}")
                predictions.append(None)
        except Exception as e:
            print(f"Error processing text: {e}")
            predictions.append(None)

    return predictions


def evaluate(predictions, labels):
    correct = sum([1 for pred, true in zip(predictions, labels) if pred == true])
    accuracy = correct / len(labels) * 100
    return accuracy


texts, labels = load_data(f'data/corpus/{dataset}.clean.txt', f'data/{dataset}.txt')

save_file = f"{dataset}_result.txt"

"""
mr range(12)
R52 range(9)
"""

def train(file_path, texts):
    for i in range(9):
        print(f'{i} cycle: ')
        new_texts = texts[i * 300:(i + 1) * 300]
        predictions = classify_text(new_texts)
        with open(file_path, 'a') as f1:
            f1.write(str(predictions))
            f1.write('\n')

def read_result(file_path):
    all_pre = []
    with open(file_path, 'r') as f2:
        res = f2.readlines()
    if dataset == 'mr':
        for lst in res:
            li = lst[1:-1].replace(',', '').split(' ')
            lst = [str(num) for num in li]
            all_pre = all_pre + lst
    else:
        for lst in res:
            actual_list = ast.literal_eval(lst)
            all_pre = all_pre + actual_list
    return all_pre


train(save_file, texts)
all_pre = read_result(save_file)

accuracy = evaluate(all_pre, labels)
print(f"Accuracy: {accuracy:.2f}%")
