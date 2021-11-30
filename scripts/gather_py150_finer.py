#!/usr/bin/python

import gzip, sys
import numpy as np


# get printable mean and std
def get_mean(x, pt=2):
    return round(np.mean(x), pt)

def get_std(x, pt=2):
    return round(np.std(x), pt)


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

    if sentence.startswith("Eval split id_val"):
        read_id_val = True
        read_id_test = False
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
    if sentence.startswith("ppl:"):
        sentence_split = sentence.split()
        acc = float(sentence_split[1])

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


    if sentence.startswith("Acc (Class-Method):"):
        sentence_split = sentence.split()
        f1 = float(sentence_split[2])

        if read_id_val:
            id_val_f1.append(f1)
    
        if read_id_test:
            id_test_f1.append(f1)
    
        if read_train:
            train_f1.append(f1)
    
        if read_val:
            val_f1.append(f1)
    
        if read_test:
            test_f1.append(f1)


input_text.close()

print('--------------------------------------------------------')

print(f"Train Ppl: {get_mean(train_acc)} ({get_std(train_acc)})      |  {train_acc}")
print(f"Train Acc (Class-Method):  {get_mean(train_f1, 3)} ({get_std(train_f1, 3)})      |  {train_f1}")
print('--------------------------------------------------------')

print(f"IID Valid Ppl: {get_mean(id_val_acc)} ({get_std(id_val_acc)})  |  {id_val_acc}")
print(f"IID Valid Acc (Class-Method):  {get_mean(id_val_f1, 3)} ({get_std(id_val_f1, 3)})  |  {id_val_f1}")

print('--------------------------------------------------------')

print(f"IID Test Ppl: {get_mean(id_test_acc)} ({get_std(id_test_acc)})   |  {id_test_acc}")
print(f"IID Test Acc (Class-Method):  {get_mean(id_test_f1, 3)} ({get_std(id_test_f1, 3)})   |  {id_test_f1}")

print('--------------------------------------------------------')

print(f"Valid Ppl: {get_mean(val_acc)} ({get_std(val_acc)})      |  {val_acc}")
print(f"Valid Acc (Class-Method):  {get_mean(val_f1, 3)} ({get_std(val_f1, 3)})      |  {val_f1}")

print('--------------------------------------------------------')

print(f"Test Ppl: {get_mean(test_acc)} ({get_std(test_acc)})       |  {test_acc}")
print(f"Test Acc (Class-Method):  {get_mean(test_f1, 3)} ({get_std(test_f1, 3)})       |  {test_f1}")

print('--------------------------------------------------------')



