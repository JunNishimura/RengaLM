import itertools
from params import *

class Preprocessor():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.vocab = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_char = ['<pad>']

    def __call__(self, sentences):
        return self.transform(sentences)

    def fit(self, sentences: list):
        '''
        build dictionaries(char_to_id & id_to_char)
        Parameters
        ----------
        sentences: list
        
        Return
        ------
        vocab: 
            sorted vocabularies
        char_to_id: dict
            dictionary(key: char, val: id)
        id_to_char: dict
            dictionary(key: id, val: char)
        '''
        # データに現れる全ての文字をソートして取得
        # 改行文字として\n, タブ区切りとして\tを追加
        self.vocab = sorted(set(list(itertools.chain.from_iterable(sentences))+['\n', '\t']))

        # 文字をキーにインデックスを返す辞書の作成
        # +1は予約語として<pad>を付け加えるため
        self.char_to_id = {v: idx+1 for idx, v in enumerate(list(self.vocab))}
        self.char_to_id['<pad>'] = 0
        
        # インデックスをキーに文字を返す辞書の作成
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
    
    def transform(self, sentences: list) -> list:
        '''
        translate the sequence of verse into the sequence of ids

        Parameters
        ----------
        sentences: list
            target text

        Return
        ------
        ids_list: list
            list of id list
        '''

        ids_list = []

        for sentence in sentences:
            sentence += '\n' # 末尾に句の終わりサインとして改行文字を入れる
            ids = [self.char_to_id[c] for c in sentence]
            ids += [0] * (self.seq_length - len(sentence)) # 0 padding. 0は<pad>に対応する
            ids_list.append(ids)

        return ids_list
    
    def decode(self, sentence_id: list) -> str:
        '''
        translate ids into the text
        Parameters
        ----------
            sentence_id: list
                the sequence of id
        
        Return
        ------
            text: str
        '''

        return ''.join([self.id_to_char[_id] for _id in sentence_id])