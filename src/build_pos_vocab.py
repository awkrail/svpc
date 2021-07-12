"""
Building pos vocab
"""
import os
import json
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def build_pos_vocab(path):
    data = load_json(path)
    pos_vocab = {}
    vocab_id = 0

    for recipe_id, val in tqdm(data.items()):
        sentences = val["sentences"]
        for sentence in sentences:
            step = nlp(sentence)
            for token in step:
                if token.pos_ == "VERB" or token.pos_ == "NOUN":
                    if token.text not in pos_vocab:
                        pos_vocab[token.text] = vocab_id
                        vocab_id += 1
    return pos_vocab

def attach_pos(path, pos_vocab):
    data = load_json(path)

    for recipe_id, val in tqdm(data.items()):
        sentences = val["sentences"]
        step_pos_words = []
        for sentence in sentences:
            step = nlp(sentence)
            pos_words = []
            for token in step:
                if token.pos_ == "VERB" or token.pos_ == "NOUN":
                    if token.text in pos_vocab:
                        pos_words.append(token.text)
            step_pos_words.append(pos_words)
        val["pos"] = step_pos_words
    return data


if __name__ == "__main__":
    data_dir = "/home/nishimura/research/recipe_generation/graph_youcook2_generator/preprocess/anet_format"
    if not os.path.exists("cache/pos_vocab_word2idx.json"):
        pos_vocab = build_pos_vocab(os.path.join(data_dir, "yc2_train_anet_format.json"))
        with open("cache/pos_vocab_word2idx.json", "w") as f:
            json.dump(pos_vocab, f)
    else:
        with open("cache/pos_vocab_word2idx.json", "r") as f:
            pos_vocab = json.load(f)
    
    filenames = ["yc2_train_anet_format.json", "yc2_val_anet_format.json"]
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        data = attach_pos(path, pos_vocab)
        with open(path, "w") as f:
            json.dump(data, f)


