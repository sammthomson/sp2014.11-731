#!/usr/bin/env python
"""
Simple translation model and language model data structures

A translation model is a dictionary where keys are tuples of French words
and values are lists of (english, log_prob) named tuples. For instance,
the French phrase "que se est" has two translations, represented like so:
tm[('que', 'se', 'est')] = [
    Phrase(english='what has', log_prob=-0.301030009985),
    Phrase(english='what has been', log_prob=-0.301030009985)]
k is a pruning parameter: only the top k translations are kept for each f.

# A language model scores sequences of English words, and must account
# for both beginning and end of each sequence. Example API usage:
lm = models.LM(filename)
sentence = "This is a test ."
lm_state = lm.begin() # initial state is always <s>
log_prob = 0.0
for word in sentence.split():
    (lm_state, word_log_prob) = lm.score(lm_state, word)
    log_prob += word_log_prob
log_prob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
"""
from UserDict import IterableUserDict
import sys
from collections import namedtuple

UNKNOWN_SYMBOL = "<unk>"
END_SYMBOL = "</s>"
START_SYMBOL = "<s>"

Phrase = namedtuple("Phrase", "english, log_prob")

NgramStats = namedtuple("NgramStats", "log_prob, backoff")


class PhraseTable(IterableUserDict):
    def __init__(self, table):
        IterableUserDict.__init__(self, table)
        self.max_foreign_len = max(len(phrase) for phrase in self.iterkeys())

    @staticmethod
    def load(filename, k):
        sys.stderr.write("Reading translation model from %s...\n" % (filename,))
        table = {}
        for line in open(filename).readlines():
            (f, e, log_prob) = line.strip().split(" ||| ")
            table.setdefault(tuple(f.split()), []).append(Phrase(e, float(log_prob)))
        for f in table:  # prune all but top k translations
            table[f].sort(key=lambda x: -x.log_prob)
            del table[f][k:]
        return PhraseTable(table)


class LanguageModel:
    def __init__(self, table):
        self.table = table

    @staticmethod
    def load(filename):
        sys.stderr.write("Reading language model from %s...\n" % (filename,))
        table = {}
        with open(filename) as in_file:
            for line in in_file:
                entry = line.strip().split("\t")
                if len(entry) > 1 and entry[0] != "ngram":
                    (log_prob, ngram, backoff) = (
                        float(entry[0]),
                        tuple(entry[1].split()),
                        float(entry[2] if len(entry) == 3 else 0.0)
                    )
                    table[ngram] = NgramStats(log_prob, backoff)
        return LanguageModel(table)

    def begin(self):
        return START_SYMBOL,

    def score(self, state, word):
        ngram = state + (word,)
        score = 0.0
        while len(ngram) > 0:
            if ngram in self.table:
                return ngram[-2:], score + self.table[ngram].log_prob
            else:  # backoff
                score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0
                ngram = ngram[1:]
        return (), score + self.table[(UNKNOWN_SYMBOL,)].log_prob

    def end(self, state):
        return self.score(state, END_SYMBOL)[1]
