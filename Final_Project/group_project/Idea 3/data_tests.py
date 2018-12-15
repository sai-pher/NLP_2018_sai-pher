import re

with open('Answers.txt') as f:
    for line in f:
        print(re.sub(r'[0-9]*\.*\t*', '', line))
