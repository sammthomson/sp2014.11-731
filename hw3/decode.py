#!/usr/bin/env python
import argparse
from itertools import islice
import sys
from models import PhraseTable, LanguageModel
import heapq
from collections import namedtuple


Hypothesis = namedtuple('Hypothesis', 'log_prob, lm_state, backpointer, phrase')
SkipKHypothesis = namedtuple('SkipKHypothesis', 'log_prob, lm_state, skipped, phrase, f, backpointer')


def extract_english(h):
    """ Follows all back-pointers to recover the full translation """
    return '' if h.backpointer is None else '%s%s ' % (
        extract_english(h.backpointer), ('' if h.phrase is None else h.phrase.english))


def extract_tm_log_prob(h):
    """ Follows all back-pointers to recover the full log probability of the given hypothesis """
    return 0.0 if h.backpointer is None else h.phrase.log_prob + extract_tm_log_prob(h.backpointer)


def get_log_prob(h):
    return h.log_prob


def pop_each(my_list):
    for i in range(len(my_list)):
        yield my_list[i], my_list[:i] + my_list[i + 1:]


class MonotoneDecoder(object):
    """ monotone decoding (doesn't permute the target phrases) """

    def __init__(self, phrase_table, language_model, max_stack_size=1, verbose=False):
        self.translation_model = phrase_table
        self.language_model = language_model
        self.max_stack_size = max_stack_size
        self.verbose = verbose

    def decode(self, foreign_sentence):
        """ Translates the given sentence into English """
        # Since decoding is monotone, all hypotheses in stacks[i] represent translations of
        # the first i words of the input sentence.
        stacks = [{} for _ in range(len(foreign_sentence) + 1)]
        stacks[0][language_model.begin()] = Hypothesis(0.0, language_model.begin(), None, None)
        for start, stack in enumerate(stacks[:-1]):
            # extend the top `stack_size` hypotheses in the current stack
            beam = heapq.nlargest(self.max_stack_size, stack.itervalues(), key=get_log_prob)
            for hypothesis in beam:
                for end in xrange(start + 1, min(start + phrase_table.max_foreign_len, len(foreign_sentence)) + 1):
                    foreign_phrase = foreign_sentence[start:end]
                    for english_phrase in phrase_table.get(foreign_phrase, ()):
                        log_prob = hypothesis.log_prob + english_phrase.log_prob
                        lm_state = hypothesis.lm_state
                        for word in english_phrase.english.split():
                            lm_state, word_log_prob = language_model.score(lm_state, word)
                            log_prob += word_log_prob
                        if end == len(foreign_sentence):
                            log_prob += language_model.end(lm_state)
                        new_hypothesis = Hypothesis(log_prob, lm_state, hypothesis, english_phrase)
                        if lm_state not in stacks[end] or stacks[end][lm_state].log_prob < log_prob:
                            stacks[end][lm_state] = new_hypothesis

        # find best translation by looking at the best scoring hypothesis
        # on the last stack
        winner = max(stacks[-1].itervalues(), key=get_log_prob)
        if self.verbose:
            tm_log_prob = extract_tm_log_prob(winner)
            sys.stderr.write('LM = %f, TM = %f, Total = %f\n' %
                             (winner.log_prob - tm_log_prob, tm_log_prob, winner.log_prob))
        return extract_english(winner)


class SkipKDecoder(object):
    """ Decoding in which up to `max_skip` phrases can be skipped and processed later """

    def __init__(self, phrase_table, language_model, max_skip=1, max_stack_size=1, verbose=False):
        self.phrase_table = phrase_table
        self.language_model = language_model
        self.max_skip = max_skip
        self.max_stack_size = max_stack_size
        self.verbose = verbose

    def decode(self, foreign_sentence):
        """ Translates the given sentence into English """
        # consider three options:
        # 1. translate an upcoming phrase
        # 2. skip an upcoming phrase
        # 3. translate one of the phrases that we previously skipped

        # all_stacks is a 3d dict:
        # all_stacks[num_processed][num_skipped][(lm_state, skipped)]
        all_stacks = [[{} for _ in range(self.max_skip + 1)] for _ in range(len(foreign_sentence) + 1)]
        all_stacks[0][0][(language_model.begin(), ())] = SkipKHypothesis(0.0,
                                                                         language_model.begin(),
                                                                         (),
                                                                         None,
                                                                         None,
                                                                         None)
        for start, ns_stack in enumerate(all_stacks[:-1]):
            # look at most-skipped first, because translating a previously skipped phrase can change
            # num_skipped without changing end.
            for num_skipped, stack in reversed(list(enumerate(ns_stack))):
                # extend the top `stack_size` hypotheses in the current stack
                beam = heapq.nlargest(self.max_stack_size, stack.itervalues(), key=get_log_prob)
                for hypothesis in beam:

                    # 1. translate an upcoming phrase
                    max_end = min(start + phrase_table.max_foreign_len, len(foreign_sentence)) + 1
                    for end in xrange(start + 1, max_end):
                        foreign_phrase = foreign_sentence[start:end]
                        for english_phrase in phrase_table.get(foreign_phrase, ()):
                            log_prob = hypothesis.log_prob + english_phrase.log_prob
                            lm_state = hypothesis.lm_state
                            for word in english_phrase.english.split():
                                lm_state, word_log_prob = language_model.score(lm_state, word)
                                log_prob += word_log_prob
                            if end == len(foreign_sentence) and num_skipped == 0:
                                log_prob += language_model.end(lm_state)
                            new_hypothesis = SkipKHypothesis(log_prob,
                                                             lm_state,
                                                             hypothesis.skipped,
                                                             english_phrase,
                                                             foreign_phrase,
                                                             hypothesis)
                            new_stack = all_stacks[end][len(new_hypothesis.skipped)]
                            new_key = (lm_state, new_hypothesis.skipped)
                            if new_key not in new_stack or new_stack[new_key].log_prob < log_prob:
                                # print('translating: %s %s' % (end, new_hypothesis))
                                new_stack[new_key] = new_hypothesis

                    # 2. skip an upcoming phrase
                    if num_skipped < self.max_skip:
                        for end in xrange(start + 1, max_end):
                            foreign_phrase = foreign_sentence[start:end]
                            for english_phrase in phrase_table.get(foreign_phrase, ()):
                                lm_state = hypothesis.lm_state
                                log_prob = hypothesis.log_prob + english_phrase.log_prob
                                skipped = hypothesis.skipped + ((english_phrase, foreign_phrase),)
                                new_hypothesis = SkipKHypothesis(log_prob,
                                                                 lm_state,
                                                                 skipped,
                                                                 None,
                                                                 None,
                                                                 hypothesis)
                                new_stack = all_stacks[end][len(new_hypothesis.skipped)]
                                new_key = (lm_state, new_hypothesis.skipped)
                                if new_key not in new_stack or new_stack[new_key].log_prob < log_prob:
                                    # print('skipping: %s %s' % (end, new_hypothesis))
                                    new_stack[new_key] = new_hypothesis

                    # 3. translate one of the phrases that we previously skipped
                    for (english_phrase, foreign_phrase), remaining in pop_each(hypothesis.skipped):
                        lm_state = hypothesis.lm_state
                        log_prob = hypothesis.log_prob
                        for word in english_phrase.english.split():
                            lm_state, word_log_prob = language_model.score(lm_state, word)
                            log_prob += word_log_prob
                        if start == len(foreign_sentence) and len(remaining) == 0:
                            log_prob += language_model.end(lm_state)
                        new_hypothesis = SkipKHypothesis(log_prob,
                                                         lm_state,
                                                         remaining,
                                                         english_phrase,
                                                         foreign_phrase,
                                                         hypothesis)
                        new_stack = all_stacks[start][len(new_hypothesis.skipped)]
                        new_key = (lm_state, new_hypothesis.skipped)
                        if new_key not in new_stack or new_stack[new_key].log_prob < log_prob:
                            # print('translating skipped: %s %s' % (end, new_hypothesis))
                            new_stack[new_key] = new_hypothesis
        # find best translation by looking at the best scoring hypothesis
        # on the last stack
        winner = max(all_stacks[-1][0].itervalues(), key=get_log_prob)
        # print(winner)
        if self.verbose:
            tm_log_prob = extract_tm_log_prob(winner)
            sys.stderr.write('LM = %f, TM = %f, Total = %f\n' %
                             (winner.log_prob - tm_log_prob, tm_log_prob, winner.log_prob))
        return extract_english(winner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
    parser.add_argument('-i', '--input', dest='input', default='data/input',
                        help='File containing sentences to translate (default=data/input)')
    parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm',
                        help='File containing translation model (default=data/tm)')
    parser.add_argument('-b', '--beam-size', dest='beam_size', default=1, type=int,
                        help='Maximum beam size (default=1)')
    parser.add_argument('-s', '--max-skips', dest='max_skip', default=0, type=int,
                        help='Maximum number of phrase that can be skipped (default=0)')
    parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int,
                        help='Number of sentences to decode (default=no limit)')
    parser.add_argument('-l', '--language-model', dest='lm', default='data/lm',
                        help='File containing ARPA-format language model (default=data/lm)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                        help='Verbose mode (default=off)')
    opts = parser.parse_args()

    phrase_table = PhraseTable.load(opts.tm, sys.maxint)
    language_model = LanguageModel.load(opts.lm)
    sys.stderr.write('Decoding %s...\n' % (opts.input,))
    if opts.max_skip > 0:
        decoder = SkipKDecoder(phrase_table,
                               language_model,
                               opts.max_skip,
                               opts.beam_size,
                               opts.verbose)
    else:
        decoder = MonotoneDecoder(phrase_table,
                                  language_model,
                                  opts.beam_size,
                                  opts.verbose)
    with open(opts.input) as input_file:
        for line in islice(input_file, opts.num_sents):
            foreign_sentence = tuple(line.strip().split())
            translation = decoder.decode(foreign_sentence)
            sys.stderr.write('.')
            print translation
