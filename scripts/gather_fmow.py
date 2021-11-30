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
train_region = []
train_year = []

id_val_acc = []
id_val_region = []
id_val_year = []

id_test_acc = []
id_test_region = []
id_test_year = []

val_acc = []
val_region = []
val_year = []

test_acc = []
test_region = []
test_year = []

# indicate which var to read
read_id_val = False
read_id_test = False
read_val = False
read_test = False
read_train = False

sub_acc_first = True

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
        sub_acc_first = True

    if sentence.startswith("Eval split id_test"):
        read_id_val = False
        read_id_test = True
        read_train = False
        read_val = False
        read_test = False
        sub_acc_first = True

    if sentence.startswith("Eval split id_val"):
        read_id_val = True
        read_id_test = False
        read_train = False
        read_val = False
        read_test = False
        sub_acc_first = True

    if sentence.startswith("Eval split test"):
        read_id_val = False
        read_id_test = False
        read_train = False
        read_val = False
        read_test = True
        sub_acc_first = True

    if sentence.startswith("Eval split val"):
        read_id_val = False
        read_id_test = False
        read_train = False
        read_val = True
        read_test = False
        sub_acc_first = True

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


    if sentence.startswith("Worst-group acc:"):
        sentence_split = sentence.split()
        f1 = float(sentence_split[2])

        if sub_acc_first:  # Year
            if read_id_val:
                id_val_year.append(f1)

            if read_id_test:
                id_test_year.append(f1)

            if read_train:
                train_year.append(f1)

            if read_val:
                val_year.append(f1)

            if read_test:
                test_year.append(f1)
            
            sub_acc_first = False

        else:  # Region
            if read_id_val:
                id_val_region.append(f1)
        
            if read_id_test:
                id_test_region.append(f1)
        
            if read_train:
                train_region.append(f1)
        
            if read_val:
                val_region.append(f1)
        
            if read_test:
                test_region.append(f1)

input_text.close()

print('--------------------------------------------------------')

print(f"Train Acc:         {get_mean(train_acc)} ({get_std(train_acc)})      |  {train_acc}")
print(f"Train min Region:  {get_mean(train_region)} ({get_std(train_region)})      |  {train_region}")
print(f"Train min Year:    {get_mean(train_year)} ({get_std(train_year)})      |  {train_year}")

print('--------------------------------------------------------')

print(f"IID Valid Acc:         {get_mean(id_val_acc)} ({get_std(id_val_acc)})  |  {id_val_acc}")
print(f"IID Valid min Region:  {get_mean(id_val_region)} ({get_std(id_val_region)})  |  {id_val_region}")
print(f"IID Valid min Year:    {get_mean(id_val_year)} ({get_std(id_val_year)})  |  {id_val_year}")

print('--------------------------------------------------------')

print(f"IID Test Acc:     {get_mean(id_test_acc)} ({get_std(id_test_acc)})   |  {id_test_acc}")
print(f"IID Test Region:  {get_mean(id_test_region)} ({get_std(id_test_region)})   |  {id_test_region}")
print(f"IID Test Year:    {get_mean(id_test_year)} ({get_std(id_test_year)})   |  {id_test_year}")

print('--------------------------------------------------------')

print(f"Valid Acc:     {get_mean(val_acc)} ({get_std(val_acc)})      |  {val_acc}")
print(f"Valid Region:  {get_mean(val_region)} ({get_std(val_region)})      |  {val_region}")
print(f"Valid Year:    {get_mean(val_year)} ({get_std(val_year)})      |  {val_year}")

print('--------------------------------------------------------')

print(f"Test Acc:      {get_mean(test_acc)} ({get_std(test_acc)})       |  {test_acc}")
print(f"Test Region:   {get_mean(test_region)} ({get_std(test_region)})       |  {test_region}")
print(f"Test Year:     {get_mean(test_year)} ({get_std(test_year)})       |  {test_year}")


print('--------------------------------------------------------')



