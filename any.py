import os
from os.path import join as PJ
import json

DATASET = "nus_wide"

root = PJ('.', 'runs_multi', DATASET)
exp_names = os.listdir(root)

data = {n: None for n in exp_names}
for n in exp_names:
    with open(PJ(root, n, "test_table.txt"), "r") as f:
        data[n] = json.load(f)

for i in range(10):
    print(data['traintest_sgd_fast0tag'][str(i + 1)]['test']['miap'])
print()
for i in range(10):
    print(data['traintest_two_layer'][str(i + 1)]['test']['miap'])
