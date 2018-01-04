import numpy as np
import argparse
from SeqGAN.SeqGAN import SeqGAN
from data import NottinghamDataloader, LSTMDataloader


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of SeqGAN")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--log_generation', type=bool, default=False)
    parser.add_argument('--total_epochs', type=int, default=1000)
    # Dataset
    parser.add_argument('--dataset', choices=['LSTM', 'Nottingham'], default='LSTM')  # 'Nottingham')
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
        args = LSTMDataloader(args.batch_size).export(args)
    elif args.dataset == 'Nottingham':
        args = NottinghamDataloader().export(args)

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
        d_filter_sizes=[3, 5, 5, 5],
        d_num_filters=[50, 80, 80, 100],
        log_generation=args.log_generation,
    )

    if args.is_train:
        model.train(sampler=args.sampler,
                    evaluator=args.evaluator,
                    evaluate=args.evaluate)


if __name__ == '__main__':
    args = argsparser()
    main(args)
