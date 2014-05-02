#!/usr/bin/env python
# written by Adam Lopez
import optparse
import sys
import bleu
from rerank import read_hypotheses


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-r", "--reference", dest="reference", default="data/dev.ref",
                         help="Target language reference sentences")
    optparser.add_option("-t", "--hypotheses", dest="hypotheses", default="data/dev.100best",
                         help="Target language hypothesis sentences")
    (opts, _) = optparser.parse_args()

    ref = [line.strip().split() for line in open(opts.reference)]
    hyps = read_hypotheses(open(opts.hypotheses))

    stats = [0 for i in xrange(10)]
    for (r, hs) in zip(ref, hyps):
        for sent_ud, h, feats in hs:
            stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h, r))]
            print bleu.bleu(stats)
