import keras
import data_utils
from keras.models import load_model
import tensorflow as tf
#/home/user/TCY/holstep
#/Users/chenyangtang/python/holstep
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_dir',
                           '/home/user/TCY/holstep',
                           'Directory where the raw data is located.')
tf.app.flags.DEFINE_string('logdir',
                           '/tmp/hol',
                           'Base directory for saving models and metrics.')
tf.app.flags.DEFINE_string('model_name',
                           'cnn_2x',
                           'Name of model to train.')
tf.app.flags.DEFINE_string('task_name',
                           'unconditioned_classification',
                           'Name of task to run: "conditioned_classification" '
                           'or "unconditioned_classification".')
tf.app.flags.DEFINE_string('tokenization',
                           'char',
                           'Type of statement tokenization to use: "char" or '
                           '"token".')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Size of a batch.')
tf.app.flags.DEFINE_integer('max_len', 512,
                            'Maximum length of input statements.')
tf.app.flags.DEFINE_integer('samples_per_epoch', 12800,
                            'Number of random step statements to draw for '
                            'training at each epoch.')
tf.app.flags.DEFINE_integer('val_samples', 246912,
                            'Number of (ordered) step statements to draw for '
                            'validation.')
tf.app.flags.DEFINE_integer('epochs', 40,
                            'Number of epochs to train.')
tf.app.flags.DEFINE_integer('verbose', 1,
                            'Verbosity mode (0, 1 or 2).')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Path to checkpoint to (re)start from.')
tf.app.flags.DEFINE_integer('data_parsing_workers', 4,
                            'Number of threads to use to generate input data.')



parser = data_utils.DataParser(FLAGS.source_dir,
                                       use_tokens=False,
                                       verbose=FLAGS.verbose)

conj_index, step_index, total = 0, 0, 0
epoch =100000
num = 0

model = load_model('myDiscriminator_model.h5')

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
for i in range(epoch):
    (X_val, _), (conj_index, step_index)= parser.draw_batch_of_steps_in_order(
        conj_index, step_index, 'val', 'integer', 512, 128)
    each = sum(model.predict_on_batch(X_val)/128)
    total = total + each
    num = num + 1
    #print (conj_index, step_index)
    print('the',num,'time:',each)
print('totla acc:',total/epoch)

