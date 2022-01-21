# @Time     : 2021/11/28 4:27 下午
# @Author   : yxq
# @FileName : train.py
import numpy as np

from utils import *
from config import *
import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_data, train_label = read_and_tokenize(os.path.join(path_base, "WikiQA-train.tsv"), maxlen, kagging)
dev_data, dev_label = read_and_tokenize(os.path.join(path_base, "WikiQA-dev.tsv"), maxlen, kagging)
test_data, test_label = read_and_tokenize(os.path.join(path_base, "WikiQA-test.tsv"), maxlen, kagging)

model = get_model()
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True, mode='max')

model.fit(train_data,
          train_label,
          batch_size=batch_size,
          epochs=epoch,
          validation_data=(dev_data, dev_label),
          callbacks=[checkpoint],
          )
