import tensorflow as tf
from tensorflow import keras
import time

import numpy as np
print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data('./imdb.npz',num_words=15000)
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

print("Training entries: {}, labels: {}".format(train_data.shape, train_labels.shape))
time.sleep(5)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

SENTENCE_LIMIT_SIZE = 200
vocab_size = len(word_index)
embed_size = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 20
EPOCHES = 10

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, 'UNK') for i in text])

# 这个暂时不用
def convert_text_to_token(sentence, word_to_token_map=word_index, limit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token

    @param sentence: 句子，str类型
    @param word_to_token_map: 单词到编码的映射
    @param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全

    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split()]

    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]

    return tokens

def padding_data(data_array):
    new_padding_data = []
    for row in data_array:
        if len(row ) < SENTENCE_LIMIT_SIZE:
            row.extend([0] * (SENTENCE_LIMIT_SIZE - len(row)))
        else:
            row = row[:SENTENCE_LIMIT_SIZE]
        new_padding_data.append(row)
    return new_padding_data

def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))

        x = x[shuffled_index]
        y = y[shuffled_index]

    # 统计共几个完整的batch
    n_batches = int(x.shape[0] / batch_size)

    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]

        yield x_batch, y_batch



test_decode = decode_review(train_data[20])
print("-----padding raw data----")
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=SENTENCE_LIMIT_SIZE)


test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=SENTENCE_LIMIT_SIZE)

def pre_build():
    input = tf.placeholder(tf.int32, [None, SENTENCE_LIMIT_SIZE], "input")
    targets = tf.placeholder(tf.float32, [None, 1], "target")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input, targets, keep_prob

def build_graph():
    embedding = tf.Variable(tf.truncated_normal((vocab_size, embed_size), stddev=0.01))
    w1 = tf.Variable(tf.random_normal([256, 16]))
    b1 = tf.Variable(tf.random_normal([16]))

    w2 = tf.Variable(tf.random_normal([16, 1]))
    b2 = tf.Variable(tf.random_normal([1]))

    input, targets, keep_prob = pre_build()
    embeded = tf.nn.embedding_lookup(embedding, input)

    average_pooling = tf.reduce_mean(embeded, [1])
    out1 = tf.matmul(average_pooling, w1) + b1
    out2 = tf.nn.dropout(out1, keep_prob)

    activate1 = tf.nn.relu(out2)
    logits = tf.add(tf.matmul(activate1, w2), b2)
    outputs = tf.nn.sigmoid(logits, name="outputs")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(outputs, 0.5), tf.float32), targets)
    accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))

    # 计算图,进行训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("./graphs/dnn", tf.get_default_graph())

        n_batches = int(train_data.shape[0] / BATCH_SIZE)

        for epoch in range(EPOCHES):
            total_loss = 0

            for x_batch, y_batch in get_batch(train_data, train_labels):
                _, batch_loss = sess.run([optimizer, loss],
                                         feed_dict={input: x_batch, targets: y_batch})

                total_loss += batch_loss

            # 在train上准确率
            train_corrects = sess.run(accuracy, feed_dict={input: train_data, targets: train_labels})
            train_acc = train_corrects / train_labels.shape[0]
            # dnn_train_accuracy.append(train_acc)

            # 在test上准确率
            test_corrects = sess.run(accuracy, feed_dict={input: test_data, targets: test_data})
            test_acc = test_corrects / test_data.shape[0]
            # dnn_test_accuracy.append(test_acc)

            print("Epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                        total_loss / n_batches,
                                                                                                        train_acc,
                                                                                                        test_acc))
        # 存储模型
        saver = tf.train.Saver()
        saver.save(sess, "./checkpoints/dnn")
        writer.close()


def restore_graph():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "checkpoints/dnn")

        total_correct = 0
        acc = sess.run(accuracy,
                       feed_dict={inputs: x_test,
                                  targets: y_test})
        total_correct += acc
        print("The DNN model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))


build_graph()

