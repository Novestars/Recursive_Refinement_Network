import subprocess
import os
from absl import flags
import torch
import rrn_flags

FLAGS = flags.FLAGS

device = None

def setup_gpu():
    global device

    # if index_gpu is None:
    if FLAGS.device == "auto":
        result = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'.split(' '),
                                stdout=subprocess.PIPE)
        lines = result.stdout.splitlines()
        lines = [int(line.decode('ascii')) for line in lines]  # get memory usage list, index is gpu number
        available_gpus = sorted(range(len(lines)), key=lines.__getitem__)  # sort GPU indices by lowest memory usage

        if available_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[0])  # pick GPU with lowest mem usage
            print('Auto-found free GPU idx {}'.format(available_gpus[0]))
            device = torch.device("cuda")
        else:
            print('Warning, no free GPU could be found, using cpu')
            device = torch.device("cpu")
    else:
        print('Using specified device:', FLAGS.device)
        device = torch.device(FLAGS.device)

    # else:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = str(index_gpu)
    #    print('Setting GPU idx {}'.format(index_gpu))