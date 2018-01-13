import abc
import numpy as np
import os
import tensorflow as tf
from SeqGAN.common import safe_log


class Dataloader(object):
    __metaClass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, seq_len, vocab_size, start_token, **kwargs):
        pass

    # def evaluate(self, fake_seq):
    #     # (optional) evaluate the generation
    #     pass

    @abc.abstractmethod
    def sample(self, batch_size):
        # return a batch
        pass

    @abc.abstractmethod
    def export(self, args):
        # takes an args variable and export
        # configurations to it
        pass


class NottinghamDataloader(Dataloader):
    def __init__(self, seq_len=30, vocab_size=88, start_token=0):
        import pretty_midi
        if not os.path.exists('data/Nottingham/'):
            raise Exception("Please download Nottingham dataset(http://www-labs.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip) and unzip it to `data/Nottingham/`.")

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.start_token = start_token

        dataset = np.array([
            np.clip(pretty_midi.
                    PrettyMIDI('data/Nottingham/train/'+fn).
                    get_piano_roll(1/.4).argmax(0)-20, 0, 87)
            for fn in os.listdir('data/Nottingham/train/')
        ])
        dataset = np.array(filter(lambda x: len(x) > seq_len, dataset))
        self.dataset = dataset
        self.N = len(dataset)

    def sample(self, batch_size):
        x = []
        for i in range(batch_size):
            ind = np.random.randint(self.N)
            song = self.dataset[ind]
            start = np.random.randint(song.shape[0]-self.seq_len)
            x.append(song[start:start+self.seq_len])
        x = np.array(x)
        return x

    def evaluate(self, fake_seqs, iteration=None):
        if iteration % 100 == 0:
            # generate a midi file from sequences every 100 iteartions
            import pretty_midi
            song = pretty_midi.PrettyMIDI()
            piano_program = pretty_midi.\
                instrument_name_to_program('Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program)

            span = .2
            last_note = None
            for song_ind, fake_seq in enumerate(fake_seqs):
                for ind, note in enumerate(fake_seq):
                    if note == 0:
                        continue
                    if note == last_note:
                        start = piano.notes.pop(-1).start
                    else:
                        start = ind*span
                    note = pretty_midi.Note(
                        velocity=100, pitch=note+19, start=start, end=(ind+1)*span)
                    piano.notes.append(note)

                song.instruments.append(piano)
                song.write('data/Nottingham/sample{}.mid'.format(song_ind))
            print 'Updated songs to `data/Nottingham/`' 

    def export(self, args):
        args.seq_len = self.seq_len
        args.vocab_size = self.vocab_size
        args.start_token = self.start_token
        args.sampler = self.sample
        args.evaluator = self.evaluate
        return args


class LSTMDataloader(Dataloader):
    def __init__(
        self,
        batch_size,
        seq_len=20, vocab_size=5000, start_token=0,
        seed=32,
        log_dir='nll_log'
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.batch_size = batch_size

        # model
        from tensorflow.contrib import rnn, seq2seq
        from SeqGAN.common import ThresholdHelper
        RNN = rnn.LSTMCell(50)
        embedding = tf.Variable(
            tf.random_normal([vocab_size, 20], stddev=1))
        decision_W = tf.Variable(
            tf.random_normal([50, vocab_size], stddev=1),
            name='decision_W')
        decision_b = tf.Variable(
            tf.random_normal([vocab_size], stddev=1),
            name='decision_b')
        given_tokens = tf.placeholder(
            tf.int32, shape=[batch_size, seq_len], name='given_tokens')
        start_tokens = tf.Variable(
            tf.tile([start_token], [batch_size]), name='start_tokens')

        generation_helper = ThresholdHelper(
            threshold=0,
            seq_len=seq_len,
            embedding=embedding,
            given_tokens=given_tokens,
            start_tokens=start_tokens,
            decision_variables=(decision_W, decision_b))
        decoder = seq2seq.BasicDecoder(
            cell=RNN, helper=generation_helper,
            initial_state=RNN.zero_state(batch_size, 'float32'))
        generation_output, _, _ = \
            seq2seq.dynamic_decode(
                decoder=decoder, maximum_iterations=seq_len)

        evaluation_helper = ThresholdHelper(
            threshold=seq_len,
            seq_len=seq_len,
            embedding=embedding,
            given_tokens=given_tokens,
            start_tokens=start_tokens,
            decision_variables=(decision_W, decision_b))
        decoder = seq2seq.BasicDecoder(
            cell=RNN, helper=evaluation_helper,
            initial_state=RNN.zero_state(batch_size, 'float32'))
        evaluation_output, _, _ = \
            seq2seq.dynamic_decode(
                decoder=decoder, maximum_iterations=seq_len)
        evaluation_prob = tf.nn.softmax(
            tf.tensordot(evaluation_output.rnn_output,
                         decision_W,
                         axes=[[2], [0]]) +
            decision_b[None, None, :])
        nll = -tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(
            tf.one_hot(given_tokens, vocab_size) *
            safe_log(evaluation_prob),
            -1)))

        self.fake_seq = generation_output.sample_id
        self.given_tokens = given_tokens
        self.nll = nll

        self.writer = tf.summary.FileWriter(log_dir)

    def sample(self, batch_size):
        return self.fake_seq.eval()

    def evaluate(self, fake_seqs, iteration=None):
        nlls = 0.
        nb_batch = len(fake_seqs)/self.batch_size
        for i in xrange(nb_batch):
            nlls += self.nll.eval({
                self.given_tokens: fake_seqs[i*self.batch_size:
                                             (i+1)*self.batch_size]})
        nll = nlls/nb_batch
        print 'nll:', nll

        self.writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(tag='nll', simple_value=nll),]),
                iteration)

    def export(self, args):
        args.seq_len = self.seq_len
        args.vocab_size = self.vocab_size
        args.start_token = self.start_token
        args.sampler = self.sample
        args.evaluator = self.evaluate
        return args
