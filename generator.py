import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from params import *
from preprocessor import Preprocessor
from model import RengaModel

def generate():
    # 連歌DataFrameの取得
    try:
        df = pd.read_pickle(RENGA_PKL_PATH)
    except:
        import pickle
        with open(RENGA_PKL_PATH, 'rb') as f:
            df = pickle.load(f)
    
    # 連歌リストの取得
    verse_list = df.stripped_verse.tolist()

    # preprocessor 
    prepro = Preprocessor(SEQ_LENGTH)
    prepro.fit(verse_list)

    # modelの構築
    VOCAB_SIZE = len(prepro.char_to_id)
    model = RengaModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    
    # modelのload
    model.load_state_dict(torch.load('../checkpoints/ckpt_20.pt')['model_state_dict'])
    model.eval()

    # inference
    generated_renga_list = []
    initials = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふめほまみむめもらりるれろわ'
    with torch.no_grad():
        for initial in initials:
            next_char = initial    
            states = model.initHidden(batch_size=1) # inference時のbatch sizeは1
            ku = ''

            # 句の生成
            while True:
                input_id = [[prepro.char_to_id[next_char]]]
                input_tensor = torch.tensor(input_id, device=model.device)
                logits, states = model(input_tensor, states)
                probs = F.softmax(torch.squeeze(logits)).cpu().detach().numpy()
                next_id = np.random.choice(VOCAB_SIZE, p=probs)
                next_char = prepro.id_to_char[next_id]

                # 改行が出たら俳句の完成合図
                if next_char == '\n':
                    break
                else:
                    ku += next_char

            generated_renga_list.append(ku)
            print(ku)

if __name__=="__main__":
    generate()