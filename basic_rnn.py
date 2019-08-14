# coding=utf-8
import os
import time
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "train.data", "train data path")
flags.DEFINE_string("test_data_path", "test.data", "test data path")


def process_data():
    train_data = read_from_file()
    test_data = read_from_file()


def read_from_file():
    print("---start to read dir----")
    train_pos_data_dir = "./aclImdb/train/pos"
    train_neg_data_dir = "./aclImdb/train/neg"
    pos_files = os.listdir(train_pos_data_dir)
    neg_files = os.listdir(train_neg_data_dir)
    if pos_files == None:
        print("pos_dir is None")
        return
    for i, name in enumerate(pos_files):
        print(i, name)
        if i > 100:
            return

def main():
    read_from_file()