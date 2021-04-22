import json
import sys

for line in sys.stdin:
    j = json.loads(line[:-1])
    if j['key'] == 'sentence':
        tokens = j['orig_tokens']
        print(' '.join(tokens))
