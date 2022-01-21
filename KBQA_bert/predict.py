# @Time     : 2021/11/28 5:12 下午
# @Author   : yxq
# @FileName : predict.py

from keras.models import load_model
from utils import *
from config import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

test_data, test_label = read_and_tokenize(os.path.join(path_base, "WikiQA-test.tsv"), maxlen, kagging)
model = load_model(filepath)
loss, acc = model.evaluate(test_data, test_label, verbose=1)
print(acc)
