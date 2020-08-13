#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create data files
"""

import os
import sys
import argparse
import numpy as np
import pickle
import itertools
from collections import defaultdict
import utils
import re
import json

from data import Sentence, Vocabulary
from action_dict import TopDownActionDict

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        # print(terminal, terminal_split)
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions

def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))

def get_data(args):
    pad = '<pad>'
    unk = '<unk>'
    def make_vocab(textfile, seqlength, minseqlength,lowercase, replace_num,
                   vocabsize, vocabminfreq, unkmethod, apply_length_filter=True):
        w2c = defaultdict(int)
        for tree in open(textfile, 'r'):
            tree = tree.strip()
            tags, sent, sent_lower = get_tags_tokens_lowercase(tree)

            assert(len(tags) == len(sent))
            if lowercase:
                sent = sent_lower
            if replace_num:
                sent = [utils.clean_number(w) for w in sent]
            if (len(sent) > seqlength and apply_length_filter) or len(sent) < minseqlength:
                continue

            for word in sent:
                w2c[word] += 1
        if unkmethod == 'berkeleyrule':
            berkeley_unks = set([utils.berkeley_unk_conv(w) for w, c in w2c.items()])
            specials = list(berkeley_unks)
        else:
            specials = [unk]
        if vocabminfreq:
            w2c = dict([(w, c) for w, c in w2c.items() if c >= vocabminfreq])
        elif vocabsize > 0 and len(w2c) > vocabsize:
            sorted_wc = sorted(list(w2c.items()), key=lambda x:x[1], reverse=True)
            w2c = dict(sorted_wc[:vocabsize])
        return Vocabulary(list(w2c.items()), pad, unkmethod, unk, specials)

    def get_nonterminals(textfiles):
        nts = set()
        for fn in textfiles:
            for tree in open(fn, 'r'):
                tree = tree.strip()
                nts.update(re.findall(r'(?=\(([^\s]+)\s\()', tree))
        nts = sorted(list(nts))
        print('Found nonterminals: {}'.format(nts))
        return nts

    def convert(textfile, lowercase, replace_num, seqlength, minseqlength,
                outfile, vocab, action_dict, apply_length_filter=True):
        dropped = 0
        sent_id = 0
        sentences = []
        for tree in open(textfile, 'r'):
            tree = tree.strip()
            action = get_actions(tree)
            tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
            orig_sent = sent[:]
            assert(len(tags) == len(sent))
            if lowercase:
                sent = sent_lower
            if (len(sent) > seqlength and apply_length_filter) or len(sent) < minseqlength:
                dropped += 1
                continue
            if replace_num:
                sent = [utils.clean_number(w) for w in sent]
            sent_ids = [vocab.get_id(t) for t in sent]
            action_ids = action_dict.to_id(action)
            sentences.append(Sentence(orig_sent, sent, sent_ids, tags, action, action_ids, tree))
            sent_id += 1
            if sent_id % 100000 == 0:
                print("{} sentences processed".format(sent_id))

        print(len(sentences))

        # Write output
        f = {"sentences": [s.to_dict() for s in sentences],
             "vocab": vocab.to_json_dict(),
             "nonterminals": nonterminals,
             "pad_token": pad,
             "unk_token": unk,
             "args": args.__dict__}

        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["sentences"]), dropped))
        json.dump(f, open(outfile, 'wt'))

    print("First pass through data to get nonterminals...")
    nonterminals = get_nonterminals([args.trainfile, args.valfile])
    action_dict = TopDownActionDict(nonterminals)

    if args.vocabfile != '':
        print('Loading pre-specified source vocab from ' + args.vocabfile)
        vocab = Vocabulary.load(args.vocabfile)
    else:
        print("First pass through data to get vocab...")
        vocab = make_vocab(args.trainfile, args.seqlength, args.minseqlength,
                           args.lowercase, args.replace_num, args.vocabsize, args.vocabminfreq,
                           args.unkmethod, 1)

    vocab.dump(args.outputfile + ".vocab")
    print("Vocab size: {}".format(len(vocab.i2w)))
    convert(args.testfile, args.lowercase, args.replace_num,
            0, args.minseqlength, args.outputfile + "-test.json",
            vocab, action_dict, 0)
    convert(args.valfile, args.lowercase, args.replace_num,
            args.seqlength, args.minseqlength, args.outputfile + "-val.json",
            vocab, action_dict, 0)
    convert(args.trainfile, args.lowercase, args.replace_num,
            args.seqlength,  args.minseqlength, args.outputfile + "-train.json",
            vocab, action_dict, 1)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize', help="Size of source vocabulary, constructed "
                                            "by taking the top X most frequent words. "
                                            " Rest are replaced with special UNK tokens.",
                                       type=int, default=100000)
    parser.add_argument('--vocabminfreq', help="Minimum frequency for vocab. Use this instead of "
                                                "vocabsize if > 1",
                                                type=int, default=0)
    parser.add_argument('--unkmethod', help="How to replace an unknown token.", choices=['unk', 'berkeleyrule'],
                        default='unk')
    parser.add_argument('--lowercase', help="Lower case", action='store_true')
    parser.add_argument('--replace_num', help="Replace numbers with N", action='store_true')
    parser.add_argument('--trainfile', help="Path to training data.", required=True)
    parser.add_argument('--valfile', help="Path to validation data.", required=True)
    parser.add_argument('--testfile', help="Path to test validation data.", required=True)
    # parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=16)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                            "than this are dropped.", type=int, default=300)
    parser.add_argument('--minseqlength', help="Minimum sequence length. Sequences shorter "
                                               "than this are dropped.", type=int, default=0)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str,
                        required=True)
    parser.add_argument('--vocabfile', help="If working with a preset vocab, "
                                            "then including this will ignore srcvocabsize and use the"
                                            "vocab provided here.",
                                            type = str, default='')
    # parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on  "
    #                                        "source length).",
    #                                        type = int, default = 1)
    args = parser.parse_args(arguments)
    # np.random.seed(3435)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
