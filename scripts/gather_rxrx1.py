#!/usr/bin/python

import gzip, sys
import numpy as np


# get printable mean and std
def get_mean(x, pt=1):
    return round(np.mean(x) * 100, pt)

def get_std(x, pt=1):
    return round(np.std(x) * 100, pt)


assert len(sys.argv) == 2

if sys.argv[1].endswith(".gz"):
    input_text = gzip.open(sys.argv[1])
else:
    input_text = open(sys.argv[1])

train_acc = []
train_f1 = []

id_val_acc = []
id_val_f1 = []

id_test_acc = []
id_test_f1 = []

val_acc = []
val_f1 = []

test_acc = []
test_f1 = []

# indicate which var to read
read_id_val = False
read_id_test = False
read_val = False
read_test = False
read_train = False

input_text_lines = input_text.readlines()
import sys
for i in range(0, len(input_text_lines)):
    sentence = input_text_lines[i]

    if sentence.startswith("Eval split train"):
        read_id_val = False
        read_id_test = False
        read_train = True
        read_val = False
        read_test = False

    if sentence.startswith("Eval split id_test"):
        read_id_val = False
        read_id_test = True
        read_train = False
        read_val = False
        read_test = False

    if sentence.startswith("Eval split test"):
        read_id_val = False
        read_id_test = False
        read_train = False
        read_val = False
        read_test = True

    if sentence.startswith("Eval split val"):
        read_id_val = False
        read_id_test = False
        read_train = False
        read_val = True
        read_test = False

    # end of identifying var
    if sentence.startswith("Average acc:"):
        sentence_split = sentence.split()
        acc = float(sentence_split[2])

        if read_id_val:
            id_val_acc.append(acc)
    
        if read_id_test:
            id_test_acc.append(acc)
    
        if read_train:
            train_acc.append(acc)
    
        if read_val:
            val_acc.append(acc)
    
        if read_test:
            test_acc.append(acc)

input_text.close()

print('--------------------------------------------------------')

print(f"Train Acc: {get_mean(train_acc)} ({get_std(train_acc)})      |  {train_acc}")
print('--------------------------------------------------------')

print(f"IID Test Acc: {get_mean(id_test_acc)} ({get_std(id_test_acc)})   |  {id_test_acc}")

print('--------------------------------------------------------')

print(f"Valid Acc: {get_mean(val_acc)} ({get_std(val_acc)})      |  {val_acc}")

print('--------------------------------------------------------')

print(f"Test Acc: {get_mean(test_acc)} ({get_std(test_acc)})       |  {test_acc}")

print('--------------------------------------------------------')



