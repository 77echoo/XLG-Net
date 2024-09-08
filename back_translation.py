import torch
import fairseq
import pickle
import os
import json
from tqdm.notebook import trange, tqdm
from utils import load_corpus


def translate_de(start, end, file_name, temperature=0.9):
    trans = {}
    for idx in tqdm(range(start, end)):
        trans[train_idxs[idx]] = de2en.translate(
            en2de.translate(train_text[idx], sampling=True, temperature=temperature), sampling=True, temperature=0.9)
        if idx % 500 == 0:
            with open("./dataset/" + file_name, 'wb') as f:
                pickle.dump(trans, f)
    with open("./dataset/" + file_name, 'wb') as f:
        pickle.dump(trans, f)


def translate_ru(start, end, file_name, temperature=0.9):
    trans = {}
    for idx in tqdm(range(start, end)):
        trans[train_idxs[idx]] = ru2en.translate(
            en2ru.translate(train_text[idx], sampling=True, temperature=temperature), sampling=True, temperature=0.9)
        if idx % 500 == 0:
            with open("./dataset/" + file_name, 'wb') as f:
                pickle.dump(trans, f)

    with open("./dataset/" + file_name, 'wb') as f:
        pickle.dump(trans, f)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = 'mr'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2de = en2de.cuda()
    de2en = de2en.cuda()
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2ru.cuda()
    ru2en.cuda()

    # Data Preprocess
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
    nb_node = adj.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_word = nb_node - nb_train - nb_val - nb_test
    nb_class = y_train.shape[1]

    y = torch.LongTensor((y_train + y_val + y_test).argmax(axis=1))
    label = {}
    label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train + nb_val], y[-nb_test:]

    corpus_file = './data/corpus/' + dataset + '_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        train_text = text.split('\n')

    train_text = train_text[:nb_train]
    train_idxs = [str(i) for i, v in enumerate(train_text)]

    translate_de(0, len(train_idxs), f"./mix_data/{dataset}/{dataset}_ru.pkl")
    translate_ru(0, len(train_idxs), f"./mix_data/{dataset}/{dataset}_ru.pkl")
