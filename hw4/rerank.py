#!/usr/bin/env python
import optparse
import sys


def read_hypothesis_line(line):
    sent_id, hypothesis_str, feat_str = line.split(' ||| ')
    feats = (feat.split('=') for feat in feat_str.split(" "))
    return int(sent_id), hypothesis_str.split(" "), [(k, float(v)) for k, v in feats]


def read_hypotheses(lines, hypotheses_per_sentence=100):
    hyps = [read_hypothesis_line(line) for line in lines]
    n = len(hyps) / hypotheses_per_sentence
    return [hyps[s*100:s*100+100] for s in xrange(0, n)]


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-k", "--kbest-list", dest="input", default="data/test.100best",
                         help="100-best translation lists")
    optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight")
    optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
    optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float",
                         help="Lexical translation model p_lex(f|e) weight")
    (opts, _) = optparser.parse_args()
    weights = {'p(e)': float(opts.lm),
               'p(e|f)': float(opts.tm1),
               'p_lex(f|e)': float(opts.tm2)}

    all_hyps = read_hypotheses(open(opts.input))
    num_sents = len(all_hyps) / 100
    for hyps_for_one_sent in all_hyps:
        best_score, best = (-1e300, '')
        for num, hyp, feats in hyps_for_one_sent:
            score = 0.0
            for k, v in feats:
                score += weights[k] * v
            if score > best_score:
                best_score, best = (score, hyp)
        sys.stdout.write("%s\n" % best)
