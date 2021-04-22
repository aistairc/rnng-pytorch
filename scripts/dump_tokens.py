import sys

from get_dictionary import get_tags_tokens_lowercase

if __name__ == "__main__":
    input_treebank_file = sys.argv[1]
    with open(input_treebank_file) as f_in:
        for line in f_in:
            tags, tokens, lc = get_tags_tokens_lowercase(line)
            print(' '.join(tokens))
