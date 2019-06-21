import argparse

DEFAULT_DATA_PATH = './data'


def commandline_config(**kwargs):
    parser = argparse.ArgumentParser(description='Cifar10 Training')
    parser.add_argument('--data', default='./data',
                        help='path to dataset')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--num_epochs', default=None, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=100, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--log_dir', default='./logs',
                        type=str, help="log directory")
    return parser.parse_args()

    # parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                     metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')
    # parser.add_argument('--prof', dest='prof', action='store_true',
    #                     help='Only run 10 iterations for profiling.')
    # parser.add_argument('--deterministic', action='store_true')
    # parser.add_argument('--sync_bn', action='store_true',
    #                     help='enabling apex sync BN.')

    # parser.add_argument('--opt-level', type=str)
    # parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    # parser.add_argument('--loss-scale', type=str, default=None)
