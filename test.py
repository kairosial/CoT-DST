import json
import time

# with open('./data/mw24_100p_test.json', 'r') as f:
#     test_data = json.load(f)

# print(type(test_data))
# print(len(test_data))

print(time.strftime('%y%m%d_%H%M%S'))

l = []
for i in range(100000):
    l.append(i)
    for j in range(100):
        l.append(j)

print(time.strftime('%y%m%d_%H%M%S'))