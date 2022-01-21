# @Time     : 2021/11/28 3:18 下午
# @Author   : yxq
# @FileName : utils.py

import keras.utils
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, Lambda
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.optimizers import Adam
import csv
import os
import random

path_base = './bert_cased_eng'
bert_vocab = os.path.join(path_base, 'vocab.txt')
bert_config = os.path.join(path_base, 'bert_config.json')
bert_ckpt = os.path.join(path_base, 'bert_model.ckpt')
tokenizer = Tokenizer(token_dict=bert_vocab)


def read_data(path, kagging=False):
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        spamreader = csv.reader(f, delimiter='\t', quotechar='"')  # delimiter是分隔符，quotechar是引用符，当一段话中出现分隔符的时候，用引用符将这句话括起来，就能排除歧义。
        next(spamreader, None)
        for row in spamreader:
            res = {}
            res['question'] = row[1]
            res['answer'] = row[5]
            res['label'] = row[6]
            result.append(res)
            if kagging:
                if len(result) >= 100:
                    break
        f.close()
    random.shuffle(result)
    return result


def tokenize(data, maxlen):
    token_ids, seg_ids, labels = [], [], []
    lens = []
    for d in data:
       token_id, seg_id = tokenizer.encode(first_text=d['question'], second_text=d['answer'], maxlen=maxlen)
       token_ids.append(token_id)
       seg_ids.append(seg_id)
       label = d['label']
       lens.append(len(token_id) + len(seg_id))
       labels.append(int(label))
    token_ids = sequence_padding(token_ids, length=maxlen)
    seg_ids = sequence_padding(seg_ids, length=maxlen)
    labels = keras.utils.to_categorical(labels, num_classes=2)
    return [token_ids, seg_ids], labels


def get_model():
    bert = build_transformer_model(
        config_path=bert_config,
        checkpoint_path=bert_ckpt,
        model='bert'
    )
    output = Lambda(lambda x: x[:, 0, :])(bert.output)
    output = Dense(2, activation='softmax')(output)
    model = Model(bert.inputs, output)
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    return model


def read_and_tokenize(path, maxlen, kagging=False):
    data = read_data(path, kagging)
    data, label = tokenize(data, maxlen)
    return data, label


if __name__ == '__main__':
    path = './datasets/raw/WikiQA-test.tsv'
    maxlen = 512
    data, label = read_and_tokenize(path, maxlen, kagging=False)
    print(data[0])
    print(label[0])
    print("end")
