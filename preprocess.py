import sqlite3
import pandas as pd
import numpy as np
import torch
import re
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from tqdm import tqdm


class NGramsDataset(torch.utils.data.Dataset):
    def __init__(self, token_data, text_data):
        self.token_data = token_data
        self.text_data = text_data
    def __len__(self):
        return len(self.token_data.input_ids)
    def __getitem__(self, index):
        d = {k: vs[index] for k, vs in self.token_data.items()}
        return d, self.text_data[index]


def get_ngrams(s, n=[1, 4]):
    '''
    Args
        s (str) : the string to chunk into n-grams
        n (list) : n values in 'n'-gram to return
    Returns
        ngrams (dict) : {n: list of n-grams in s}
    This function performs the following:
      - lower case the prompt and strip whitespaces
      - tokenize the string using delimiters: ;,|\W
      - forms n-grams from tokens by maintaining a sliding window of length n
        over all tokens
    '''
    s = s.lower().strip()

    delimiters = '; |, |\| | |\W|,|;|\|' # ensure delimiters are retained
    splitted = re.split(delimiters, s)
    ngrams = {}

    for cur_n in n:
        # form n-grams as sliding window of size n
        cur_ngrams = [
            ' '.join(splitted[i:i + cur_n]) 
            for i in range(len(splitted) - cur_n)
        ]
        ngrams[cur_n] = cur_ngrams

    return ngrams


def main():
    # constants
    clip_version = 'openai/clip-vit-large-patch14'
    db_path = 'custom_data/sac_public_2022_06_29.sqlite'
    ngrams_n = [4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_as_hf_dataset = False

    # read the DB
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    print('Loaded DB with tables')
    print(c.execute(
        'select name from sqlite_master where type="table"'
    ).fetchall())

    # fetch prompts from the DB
    df = pd.read_sql_query('select * from generations', conn)
    unique_prompts = df['prompt'].drop_duplicates().tolist()
    all_lengths = np.asarray(list(map(lambda x: len(x.split()), unique_prompts)))
    print('Data statistics:')
    print(f'Mean str len: {all_lengths.mean()}\nMax str len: {all_lengths.max()}')

    # preprocess prompts to ngrams
    data = {n: [] for n in ngrams_n}
    for sent in unique_prompts:
        sent_ngrams = get_ngrams(sent, n=ngrams_n)
        for n in ngrams_n:
            data[n] += sent_ngrams[n]
    data = {k: sorted(list(set(v))) for k, v in data.items()}
    print('Preprocessed data statistics:')
    for k, v in data.items():
        print(f'Number of examples for n={k}: {len(v)}')

    # clip model
    model = CLIPModel.from_pretrained(clip_version).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(clip_version)

    # get embeddings
    docs = []
    for v in data.values():
        docs += v

    model_inp = processor(text=docs, return_tensors='pt', padding=True)
    dataset = NGramsDataset(model_inp, docs)
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    
    collated_prompts, collated_reps = [], []

    for b_tok, b_str in tqdm(loader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in b_tok.items()} # (inp_ids, attn_mask)
            model_outs = model.get_text_features(**batch).cpu().tolist()
            collated_prompts += b_str
            collated_reps += model_outs

    # save data
    clip_text_embed = np.asarray(collated_reps)
    with open('ngram_prompts.txt', 'w') as f:
        for p in collated_prompts:
            f.write(p + '\n')
    np.save('clip_text_embed.npy', clip_text_embed)


if __name__ == '__main__':
    main()