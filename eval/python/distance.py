import argparse
import numpy as np
import sys
import os
import csv
import random


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    fi = open(f'runs/{random.random()}{args.vocab_file}.csv', 'w')
    fi.write('lookup, context, word, cosine distance\n')
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab, fi)


def distance(W, vocab, ivocab, input_term, fi):
    if '|' in input_term:
        lookup, input_term = input_term.split(' | ')

    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    # lookup_vec = np.copy(W[vocab[lookup], :])

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    # lookup_dist = np.dot(lookup_vec, vec_norm.T)
    # print(lookup

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.inf

    # lookup_dist = np.dot(W, lookup_vec.T)
    lookup_index = vocab[lookup]
    a = np.argsort(-dist)
    # lookup_pos = a.index(lookup_index)
    # lookup_index, = np.where(a == lookup)
    lookup_pos = np.where(a == lookup_index)[0][0] + 1
    a = a[:N]
    print('Lookup value', dist[lookup_index])
    print('Lookup index', lookup_index)
    # print('Lookup dist', lookup_dist)
    print('Lookup position', lookup_pos)
    fi.write(f'{lookup}, {input_term}, {lookup_pos}, {dist[lookup_index]}\n')
    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


if __name__ == "__main__":
    N = 10 # number of closest words that will be shown
    W, vocab, ivocab, fi = generate()
    while True:
        input_term = input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            distance(W, vocab, ivocab, input_term, fi)
