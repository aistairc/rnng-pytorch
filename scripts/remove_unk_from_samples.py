import sys
import fileinput
from strip_functional import PhraseTree
from remove_dev_unk import remove_dev_unk

def main():
    if len(sys.argv) != 4:
        raise NotImplementedError('Program only takes three arguments: the original sentence file, the output predicted tree file, and num_samples')
    num_samples = int(sys.argv[3])
    gold_lines = []
    with open(sys.argv[1], 'r') as gold_file:
        for line in gold_file:
            tokens = ['(X {})'.format(t) for t in line.strip().split()]
            gold_lines += ['(S {})'.format(' '.join(tokens))] * num_samples

    # use fileinput so we can pass "-" to read from stdin
    sys_lines = list(fileinput.input(files=[sys.argv[2]]))

    assert len(gold_lines) == len(sys_lines)
    for gold_line, sys_line in zip(gold_lines, sys_lines):
        output_string = remove_dev_unk(gold_line, sys_line)
        print(output_string.rstrip())

if __name__ == '__main__':
    main()
