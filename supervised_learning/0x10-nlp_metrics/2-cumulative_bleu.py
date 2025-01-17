#!/usr/bin/env python3
"""
Function that calculates cumulative n-gram BLEU score for a sentence
"""


import numpy as np


def ngram_bleu(references, sentence, n, n_precissions=[], reference_len=0):
    """
    calculates the unigram BLEU score for a sentence:
    Returns: the unigram BLEU score
    """
    output_len = len(sentence)
    count_clip = 0
    references_len = []
    counts_clip = {}

    if n != 0:
        n_sentence = [' '.join([str(j) for j in sentence[i:i+n]])
                      for i in range(len(sentence)-(n-1))]
        n_output_len = len(n_sentence)

        for reference in references:
            n_reference = [' '.join([str(j) for j in reference[i:i+n]])
                           for i in range(len(sentence)-(n-1))]
            references_len.append(len(reference))
            for word in n_reference:
                if word in n_sentence:
                    if not counts_clip.keys() == word:
                        counts_clip[word] = 1

        count_clip = sum(counts_clip.values())

        reference_len = min(references_len, key=lambda x: abs(x-output_len))

        precission = (np.log(count_clip/n_output_len))

        n_precissions.append(precission)
        return (ngram_bleu(references, sentence, n-1,
                           n_precissions, reference_len))

    else:
        return n_precissions, reference_len


def cumulative_bleu(references, sentence, n):
    """
    references is a list of reference translations
      each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the largest n-gram to use for evaluation
    All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score
    """
    output_len = len(sentence)

    n_precissions, reference_len = ngram_bleu(references, sentence, n)
    n_precissions = np.asarray(n_precissions)
    n_precissions /= n
    precission = np.exp(np.sum(n_precissions))

    if output_len > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / output_len))

    bleu_score = bp * precission

    return bleu_score
