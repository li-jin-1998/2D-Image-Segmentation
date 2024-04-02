import os

paths = os.listdir('data/train/image')

i = 0
for path in os.listdir('Final_test/image'):
    if path in paths:
        print(path)
        i = i + 1

print(i)
