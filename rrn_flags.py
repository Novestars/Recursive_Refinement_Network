from absl import flags


FLAGS = flags.FLAGS

# General flags.
flags.DEFINE_string('dataset', 'lung',
                    'Dataset to use, i.e. "lung".')

flags.DEFINE_bool(
    'use_minecraft_camera_actions', False, 'If true, append camera actions and xy grid augmentation to flow layers input.')

flags.DEFINE_integer('num_epochs', 500,
                     'Number epochs for training.')

flags.DEFINE_integer('batch_size', 2,
                     'Batch size for training.')

flags.DEFINE_bool('continue_training', True , 'If true, continue training at previous checkpoint, otherwise start over.')

flags.DEFINE_string('device', 'auto',
                    'Device to use, i.e. "cpu" or "cuda:0", or "auto" to automatically select best GPU')

# loss flags
flags.DEFINE_float('weight_smooth1', 1.25, 'Weight for smoothness loss.')
flags.DEFINE_float('weight_lcc', 1.00, 'Weight for LCC loss.')


#0.005    i2e
