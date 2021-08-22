import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from params import *
from preprocessor import Preprocessor
from dataset import RengaDataset
from model import RengaModel

def visualize_training(history: dict, label: str):
    result = history[label]

    fig = plt.figure(figsize=(12, 9))
    plt.plot(range(len(result)), result, linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.xticks(range(len(result)), [i*10 for i in range(len(result))])
    plt.ylabel(label)
    plt.legend()
    plt.savefig(f'{label}_result.jpg')
    plt.show()

def train():
    # 俳句DataFrameの取得
    try:
        df = pd.read_pickle(RENGA_PKL_PATH)
    except:
        import pickle
        with open(RENGA_PKL_PATH, 'rb') as f:
            df = pickle.load(f)
    
    # 俳句リストの取得
    verse_list = df.stripped_verse.tolist()

    # preprocessor 
    prepro = Preprocessor(SEQ_LENGTH)
    prepro.fit(verse_list)
    
    # haiku_textをidの列に変換
    ids_list = prepro(verse_list)

    # 俳句データセットの作成
    dataset = RengaDataset(ids_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # trainingの準備
    vocab_size = len(prepro.char_to_id)
    model = RengaModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # epochの定義
    epoch = 1

    # 可視化の有無
    isViz = True

    # checkpointの有無を確認
    checkpoint_dir = './checkpoints'
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)):
        # checkpointが最終のものを抽出して、途中から学習を再開する
        epoch = max(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(checkpoint_dir)))
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'ckpt_{epoch}.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        isViz = False
    else:
        # checkpointディレクトリを作成
        os.makedirs(checkpoint_dir, exist_ok=True)

    # train
    model.train()
    history = {'loss': [], 'ppl': []}
    while epoch <= EPOCHS:
        print('-'*25)
        print(f'EPOCH: {epoch}')

        total_loss = 0
        for X_train, y_train in dataloader:
            optimizer.zero_grad()
            X_train, y_train = X_train.to(model.device), y_train.to(model.device)
            state_h, state_c = model.initHidden(BATCH_SIZE)

            y_pred, (state_h, state_c) = model(X_train, (state_h, state_c))
            y_pred = y_pred.view(-1, vocab_size)
            y_train = y_train.view(-1)
            loss = criterion(y_pred, y_train)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        current_loss = total_loss / len(dataloader)
        current_ppl  = np.exp(current_loss)
        print(f'LOSS: {current_loss}, PERPLEXITY: {current_ppl}')
        history['loss'].append(current_loss)
        history['ppl'].append(current_ppl)

        # 10epoch毎にsave
        if epoch % 10 == 0:
            path = f'./checkpoints/ckpt_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss
            }, path)
        
        # epochの更新
        epoch += 1
    
    # 学習の最後にモデルを保存
    torch.save(model.state_dict(), f'./checkpoints/final.pt')

    # 結果を可視化する
    if isViz:
        visualize_training(history, 'loss')
        visualize_training(history, 'ppl')

if __name__ == '__main__':
    train()