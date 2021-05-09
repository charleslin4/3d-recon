DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIRECTORY = '/home/data'
SAVE_PATH = '/home/data'
BATCH_SIZE = 64 # batchsize used in Pytorch3D tech report
POINT_CLOUD_SIZE = 10000

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', help='baseline or vq-vae')
    parser.add_argument('--name', type=str, default='0', help='name of experiment')
    parser.add_argument('--bs', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--epochs', type=int, default=1)

    # Options for testing
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=SAVE_PATH, help='path to save results to')

    return parser.parse_args()