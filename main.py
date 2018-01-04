import numpy as np
import argparse
from SeqGAN.SeqGAN import SeqGAN 

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of SeqGAN")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--log_generation', type=bool, default=True)
    parser.add_argument('--total_epochs', type=int, default=1000)
    # Dataset
    parser.add_argument('--dataset', choices=['LSTM', 'Nottingham'], default='Nottingham')
    parser.add_argument('--batch_size', type=int, default=32)
    # SeqGAN 
    # Generator
    parser.add_argument('--g_emb_dim', type=int, default=32)
    parser.add_argument('--g_hidden_dim', type=int, default=32)
    # Discriminator
    parser.add_argument('--d_emb_dim', type=int, default=32)
    args = parser.parse_args()
    return args


def main(args):
    np.random.seed(args.seed)
    if args.dataset == 'LSTM':
        args.seq_len = 32
        args.vocab_size = 50
        args.start_token = 0
    elif args.dataset == 'Nottingham':
        args.seq_len = 50
        args.vocab_size = 88
        args.start_token = 0

        import pretty_midi
        import os
        dataset = np.array([
            np.clip(pretty_midi.
                    PrettyMIDI('data/Nottingham/train/'+fn).
                    get_piano_roll(1/.4).argmax(0)-20, 0, 87)
            for fn in os.listdir('data/Nottingham/train/')
        ])
        dataset = np.array(filter(lambda x: len(x) > args.seq_len, dataset))

        def sampler(bs):
            x = []
            for i in range(bs):
                ind = np.random.randint(len(dataset))
                song = dataset[ind]
                start = np.random.randint(song.shape[0]-args.seq_len)
                x.append(song[start:start+args.seq_len])
            x = np.array(x)
            return x
        args.sampler = sampler

    model = SeqGAN(
        start_token=args.start_token,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        # g
        g_emb_dim=args.g_emb_dim,
        g_hidden_dim=args.g_hidden_dim,
        # d
        d_emb_dim=args.d_emb_dim,
        d_filter_sizes=[3, 5, 5],
        d_num_filters=[50, 80, 80],
        log_generation=args.log_generation,
    )

    if args.is_train:
        model.train(args.sampler)


if __name__ == '__main__':
    args = argsparser()
    main(args)
