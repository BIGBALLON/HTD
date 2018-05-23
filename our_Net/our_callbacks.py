import argparse
import numpy as np
import tensorflow as tf
import math
from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback

learning_rate_scheduler_list = [
'step_decay', 
'exponential', 'two_stage_exponential',
'tanh_restart', 'cos_restart',
'tanh', 'cos', 'tanh_iteration', 'cos_iteration'
]

class ModelCheckpointWithEpoch(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, save_begin_epoch=100):
        super(ModelCheckpointWithEpoch, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.save_begin_epoch = save_begin_epoch

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        elif epoch > self.save_begin_epoch:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class TensorBoardWithLr(TensorBoard):
    def __init__(self, log_dir='./logs',
             histogram_freq=0,
             batch_size=32,
             write_graph=True,
             write_grads=False,
             write_images=False,
             embeddings_freq=0,
             embeddings_layer_names=None,
             embeddings_metadata=None):
        super(TensorBoardWithLr, self).__init__(log_dir,
                                             histogram_freq,
                                             batch_size,
                                             write_graph,
                                             write_grads,
                                             write_images,
                                             embeddings_freq,
                                             embeddings_layer_names,
                                             embeddings_metadata)


    def on_train_begin(self, logs=None):
        self.opt = self.model.optimizer
        self.opt_name = type(self.opt).__name__
        self.lr = self.opt.lr

    def on_batch_end(self, batch, logs=None):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = K.get_value(self.lr)
        summary_value.tag = 'real_lr'
        self.writer.add_summary(summary, K.get_value(self.opt.iterations))
        self.writer.flush() 

    def on_epoch_end(self, epoch, logs=None):
        super(TensorBoardWithLr, self).on_epoch_end(epoch, logs)



class LearningRateScheduler(Callback):
    def __init__(self, args, iterations, learning_rate_scheduler):
        super(LearningRateScheduler, self).__init__()
        if (args.learning_rate_method in learning_rate_scheduler_list) == False:
            print("[ERROR] no method ", args.learning_rate_method)
            exit()
        self.args = args
        self.method = args.learning_rate_method
        self.learning_rate_scheduler = learning_rate_scheduler
        self.T_e    = 10.
        self.T_mul  = 2.
        self.T_next = self.T_e
        self.tt     = 0
        self.start_lr=self.learning_rate_scheduler[0][0]
        self.end_lr=self.learning_rate_scheduler[0][-1]
        self.iterations=iterations

    def on_train_begin(self, log=None):
        self.opt = self.model.optimizer

    def on_batch_begin(self, batch, log):
        lr = K.get_value(self.opt.lr)
        iteration = K.get_value(self.opt.iterations)*1.

        if self.method == "cos_iteration":
            lr = (self.start_lr+self.end_lr)/2.+(self.start_lr-self.end_lr)/2.*math.cos(math.pi/2.*(iteration/(self.iterations*self.args.epochs/2.)))
        elif self.method == 'tanh_iteration':
            lr = (self.start_lr+self.end_lr)/2. - (self.start_lr-self.end_lr)/2. * math.tanh(8.*iteration/(self.iterations*self.args.epochs) - 4.)
        elif self.method == 'cos_restart':
            # cos without shift
            dt = math.pi/float(self.T_e)
            self.tt = self.tt+float(dt)/self.iterations
            if self.tt >= math.pi:
                self.tt = self.tt - math.pi
            lr = self.end_lr + 0.5*(self.start_lr - self.end_lr)*(1+ math.cos(self.tt))
        elif self.method == 'tanh_restart':
            # tanh restart
            dt = 1./(self.T_e*self.iterations)
            self.tt = self.tt+dt
            lr = (self.start_lr+self.end_lr)/2. - (self.start_lr-self.end_lr)/2. * math.tanh(8.*self.tt - 4.)
     
        K.set_value(self.opt.lr, lr)

    def on_epoch_begin(self, epoch, log):
        lr = K.get_value(self.opt.lr)
        if self.method == "step_decay":
            for i in range(len(self.learning_rate_scheduler[1])):
                if self.learning_rate_scheduler[1][i] <= epoch < self.learning_rate_scheduler[1][i+1]:
                    lr = self.learning_rate_scheduler[0][i]
        elif self.method == "exponential":
            lr = self.start_lr*(0.98**epoch)
        elif self.method == "cos":
            lr = (self.start_lr+self.end_lr)/2.+(self.start_lr-self.end_lr)/2.*math.cos(math.pi/2.*(epoch/(self.args.epochs/2.)))
        elif self.method == 'tanh':
            start = self.args.tanh_begin
            end = self.args.tanh_end
            lr = self.start_lr / 2. * ( 1- math.tanh( (end-start)*epoch/self.args.epochs + start))
        elif self.method == 'two_stage_exponential':
        	if epoch <= 100:
        		lr = self.start_lr*(0.995**epoch)
        	else:
        		lr = self.start_lr*(0.995**100)*(0.96**(epoch-100))
        K.set_value(self.opt.lr, lr)

        if(epoch+1 == self.T_next):
            self.tt = 0
            self.T_e = self.T_e*self.T_mul
            self.T_next = self.T_next + self.T_e
