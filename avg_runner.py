import tensorflow as tf
import numpy as np
import getopt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from glob import glob
from utils import get_train_batch, get_test_batch
import constants as c
from g_model import GeneratorModel
from d_model import DiscriminatorModel


class AVGRunner:
    def __init__(self, num_steps, model_load_path):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_steps: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        """
        self.global_step = 0
        self.num_steps = num_steps

        self.sess = tf.Session()
        #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        if c.ADVERSARIAL:
            print('Init discriminator...')
            self.d_model = DiscriminatorModel(self.sess,
                                              self.summary_writer,
                                              c.TRAIN_HEIGHT,
                                              c.TRAIN_WIDTH,
                                              c.SCALE_CONV_FMS_D,
                                              c.SCALE_KERNEL_SIZES_D,
                                              c.SCALE_FC_LAYER_SIZES_D)

        print('Init generator...')
        self.g_model = GeneratorModel(self.sess,
                                      self.summary_writer,
                                      c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH,
                                      c.FULL_HEIGHT,
                                      c.FULL_WIDTH,
                                      c.SCALE_FMS_G,
                                      c.SCALE_KERNEL_SIZES_G)

        print('Init variables...')
        self.summary_writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

        # if load path specified, load a saved model
        if model_load_path is not None:
            self.saver.restore(self.sess, model_load_path)
            print('Model restored from ' + model_load_path)
            
        

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        
        np.random.shuffle(c.TEST_EXAMPLES)
        np.random.shuffle(c.TRAIN_EXAMPLES)
        
        examples_count = 0
        num_epoch = 0
        print('EPOCH - ' + str(num_epoch))
        for i in range(self.num_steps):
            
            
            
            if c.ADVERSARIAL : 
                # update discriminator
                batch = get_train_batch(examples_count)
                #print('Training discriminator...')
                self.d_model.train_step(batch, self.g_model)

            # update generator
            batch = get_train_batch(examples_count)
            
            examples_count += c.BATCH_SIZE
            
            #print('Training generator...')
            self.global_step = self.g_model.train_step(
                batch, discriminator=(self.d_model if c.ADVERSARIAL else None))

            #test batch each 'epoch'
            
            if examples_count >= c.NUM_CLIPS:
                np.random.shuffle(c.TRAIN_EXAMPLES)
                examples_count = 0
                self.test(c.TEST_BATCH_SIZE, full=True)#bsize = c.NUM_TEST_CLIPS,full=True)
                num_epoch += 1
                print('EPOCH - ' + str(num_epoch))

            # save the models
            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('-' * 30)
                print('Saving models...')
                self.saver.save(self.sess,
                                c.MODEL_SAVE_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print('Saved models!')
                print('-' * 30)

            # test generator model
            #if self.global_step % c.TEST_FREQ == 0:
            #    self.test()

    def test(self, bsize = c.BATCH_SIZE, full=False):
        """
        Runs one test step on the generator network.
        """
        
        '''
        batch = get_test_batch(c.BATCH_SIZE)
        '''
        
        batch = np.empty([bsize, c.FULL_HEIGHT, c.FULL_WIDTH, (3 * (c.HIST_LEN + 1))],
                     dtype=np.float32)
        
        if full:
            # can be very memory hungry
            if c.TEST_CLIPS_FULL.size == 0:
                c.TEST_CLIPS_FULL = np.empty([c.NUM_TEST_CLIPS, c.FULL_HEIGHT, c.FULL_WIDTH, (3 * (c.HIST_LEN + 1))],
                     dtype=np.float32)
                for i in range(c.NUM_TEST_CLIPS):
                    path = c.TEST_EXAMPLES[i]
                    clip = np.load(path)['arr_0']
                    c.TEST_CLIPS_FULL[i] = clip
            
            offset = np.random.choice(np.arange(c.NUM_TEST_CLIPS - bsize))
            batch = c.TEST_CLIPS_FULL[offset:(offset+bsize),:,:,:]
            
        else:
            offset = np.random.choice(np.arange(c.NUM_TEST_CLIPS - bsize))
            for i in range(bsize):
                #path = c.TEST_DIR + str(np.random.choice(c.NUM_TEST_CLIPS)) + '.npz'
                path = c.TEST_EXAMPLES[offset+i]
                clip = np.load(path)['arr_0']
                batch[i] = clip
                        
        self.g_model.test_batch(
            batch, self.global_step)
        
    #def make_video(self, source, dest=None, double_framerate=True)


def usage():
    print('Options:')
    print('-l/--load_path=    <Relative/path/to/saved/model>')
    print('-t/--test_dir=     <Directory of test images>')
    print('-c/--clips_dir=     <Directory of training clips. Default=../Data/.Clips>')
    print('-a/--adversarial=  <{t/f}> (Whether to use adversarial training. Default=True)')
    print('-n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>')
    print('-s/--steps=        <Number of training steps to run> (Default=1000001)')
    print('-O/--overwrite     (Overwrites all previous data for the model with this save name)')
    print('-T/--test_only     (Only runs a test step -- no training)')
    print('-H/--help          (Prints usage)')
    print('--stats_freq=      <How often to print loss/train error stats, in # steps>')
    print('--summary_freq=    <How often to save loss/error summaries, in # steps>')
    print('--img_save_freq=   <How often to save generated images, in # steps>')
    print('--test_freq=       <How often to test the model on test data, in # steps>')
    print('--model_save_freq= <How often to save the model, in # steps>')


def main():
    ##
    # Handle command line input.
    ##

    load_path = None
    test_only = False
    num_steps = 1000001
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'l:t:r:a:n:s:c:g:OTH',
                                ['load_path=', 'test_dir=', 'adversarial=', 'name=',
                                 'steps=', 'overwrite', 'test_only', 'help', 'stats_freq=',
                                 'summary_freq=', 'img_save_freq=', 'test_freq=',
                                 'model_save_freq=', 'clips_dir=', 'adv_w=', 'lp_w=', 'gdl_w=' , 'batch_size=' , 'lrateG=', 'lrateD='] )
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-l', '--load_path'):
            load_path = arg
        if opt in ('-t', '--test_dir'):
            c.TEST_DIR = arg
            c.NUM_TEST_CLIPS = len(glob(c.TEST_DIR + '*.npz'))
            c.TEST_EXAMPLES = np.array(glob(c.TEST_DIR + '*.npz'))
            if c.NUM_TEST_CLIPS>0:
                path = c.TEST_DIR + '0.npz'
                clip = np.load(path)['arr_0']
                c.FULL_HEIGHT = clip.shape[0]
                c.FULL_WIDTH  = clip.shape[1]
            #c.set_test_dir(arg)
        if opt in ('-a', '--adversarial'):
            c.ADVERSARIAL = (arg.lower() == 'true' or arg.lower() == 't')
        if opt in ('-n', '--name'):
            c.set_save_name(arg)
        if opt in ('-s', '--steps'):
            num_steps = int(arg)
        if opt in ('-O', '--overwrite'):
            c.clear_save_name()
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)
        if opt in ('-T', '--test_only'):
            test_only = True
        if opt == '--stats_freq':
            c.STATS_FREQ = int(arg)
        if opt == '--summary_freq':
            c.SUMMARY_FREQ = int(arg)
        if opt == '--img_save_freq':
            c.IMG_SAVE_FREQ = int(arg)
        if opt == '--test_freq':
            c.TEST_FREQ = int(arg)
        if opt == '--model_save_freq':
            c.MODEL_SAVE_FREQ = int(arg)
        if opt in ('-c', '--clips_dir'):
            c.TRAIN_DIR_CLIPS = arg
            c.NUM_CLIPS = len(glob(c.TRAIN_DIR_CLIPS + '*.npz'))
            c.TRAIN_EXAMPLES = np.array(glob(c.TRAIN_DIR_CLIPS + '*.npz'))
        if opt in ('--adv_w'):
            c.LAM_ADV = float(arg)
        if opt in ('--lp_w'):
            c.LAM_LP = float(arg)
        if opt in ('--gdl_w'):
            c.LAM_GDL = float(arg)
        if opt in ('--batch_size'):
            c.BATCH_SIZE = int(arg)
        if opt in ('--lrateG'):
            c.LRATE_G = float(arg)
        if opt in ('--lrateD'):
            c.LRATE_D = float(arg)
            
    # set test frame dimensions
    #assert os.path.exists(c.TEST_DIR)
    #c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()

    ##
    # Init and run the predictor
    ##
    
    runner = AVGRunner(num_steps, load_path)
    if test_only:
        runner.test()
    else:
        runner.train()


if __name__ == '__main__':
    main()
