import argparse
import scipy.io


from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from time import time
from preprocessing import *
from utils import *
parser = argparse.ArgumentParser(description='Experiments for CIFAR10')
## general arguments
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='the dataset on which to execute the program')
parser.add_argument('--traintime', type=bool, default=False, metavar='T',
                    help='True at train time, False at test time ')
parser.add_argument('--model-id', type=int, default=0, metavar='MID',
                    help='the id of the trained model')
parser.add_argument('--coarse-rate', type=int, default=100, metavar='C',
                    help='100 times the proportion of coarse labels')
parser.add_argument('--middle-rate', type=int, default=60, metavar='M',
                    help='100 times the proportion of fine labels')
parser.add_argument('--fine-rate', type=int, default=20, metavar='F',
                    help='100 times the proportion of fine labels')
parser.add_argument('--perturbation', type=str, default='hide_top', metavar='P',
                    help='the perturbation to add to the test set')
parser.add_argument('--s', type=float, default=2, metavar='S',
                    help='the smoothing parameter of the distortion')
parser.add_argument('--t', type=float, default=0.5, metavar='T',
                    help='the intensity parameter of the distortion')
## arguments for the training process
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='the number of epochs')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='LR',
                    help='input batch learning rate for training (default: 1e-3)')
parser.add_argument('--opt', type=str, default='Adam', metavar='O',
                    help='the chosen optimizer for training')
parser.add_argument('--FT', type=bool, default=False, metavar='FT',
                    help='True if fine tuning the fine classification')
parser.add_argument('--single', type=bool, default=False, metavar='SI',
                    help='True if training the single model network')

args = parser.parse_args()
id = args.model_id
c = args.coarse_rate
m = args.middle_rate
f = args.fine_rate
s = args.s
t = args.t
## defining the models according to the chosen dataset

if args.dataset == 'mnist' :
    from models_mnist import *
    coarse_model = Mnist_classifier_coarse()
    middle_model = Mnist_classifier_middle()
    fine_model = Mnist_classifier_fine()
    single_model = Mnist_classifier_full()
elif args.dataset == 'fashion_mnist' :
    from models_fashion_mnist import *
    coarse_model = FashionMnist_classifier_coarse()
    middle_model = FashionMnist_classifier_middle()
    fine_model = FashionMnist_classifier_fine()
    single_model = FashionMnist_classifier_full()

if args.dataset == 'cifar10' :
    from models_cifar10 import *
    coarse_model = coarse_modelUnet_AP()
    middle_model = middle_modelUnet_AP()
    fine_model = fine_modelUnet_AP()
    single_model = fine_modelUnet_single_AP()

if args.dataset == 'SVHN' :
    from models_SVHN import *
    coarse_model = coarse_modelUnet()
    middle_model = middle_modelUnet()
    fine_model = fine_modelUnet()
    single_model = fine_modelUnet_single()

## defining the datasets
x_trains,x_vals,x_test,y_trains,y_vals,y_test = data_processing(c/100, m/100, f/100,args.perturbation, s, t, args.dataset)

if args.traintime == True :
    logdir1 = 'log_dir/logcoarse{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir2 = 'log_dir/logmiddle{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir3 = 'log_dir/logfine{}_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir4 = 'log_dir/logfine{}_tuning_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    logdir5 = 'log_dir/logfine{}_single_id{}c{}m{}f{}'.format(args.dataset,id,c, m,f)
    import os
    if not os.path.exists(logdir1):
        os.makedirs(logdir1)
    if not os.path.exists(logdir2):
        os.makedirs(logdir2)
    if not os.path.exists(logdir3):
        os.makedirs(logdir3)
    if not os.path.exists(logdir4):
        os.makedirs(logdir4)
    if not os.path.exists(logdir5):
        os.makedirs(logdir5)
    if args.opt =='Adam':
        optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    if args.opt =='SGD':
        optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)
    if args.opt =='RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(lr=args.learning_rate, decay = 1e-6)
    if not(args.single):
        if not(args.FT):
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir1)]
            coarse_model.compile(loss = {"coarse": 'categorical_crossentropy'},
                loss_weights = {"coarse":1.},optimizer= optimizer ,metrics=['accuracy'])
            coarse_model.fit(x_trains[0], y_trains[0],
                      batch_size=args.batch_size,
                      epochs=20,
                      verbose=1,
                      callbacks = callbacks,
                      validation_data=(x_vals[0], y_vals[0]))
            coarse_model.save_weights('weights/model_multi_outputcoarse{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f))




            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.4,patience =5),
            TensorBoard(log_dir = logdir2)]
            middle_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy'},
               loss_weights = {"coarse":1.,"middle":float(c)/m},optimizer= optimizer ,metrics=['accuracy'])
            middle_model.load_weights('weights/model_multi_outputcoarse{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f),by_name = True)
            middle_model.fit(x_trains[1], y_trains[1],
                     batch_size=args.batch_size,
                     epochs=20,
                     verbose=1,
                     callbacks = callbacks,
                     validation_data=(x_vals[1], y_vals[1]))
            middle_model.save_weights('weights/model_multi_outputmiddle{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f))


            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir3),
            EarlyStopping(monitor='val_loss',patience=7, verbose=0, mode='auto', baseline=None, restore_best_weights=True)]
            fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine":
                                         'categorical_crossentropy'},
                loss_weights = {"coarse":1.,"middle":float(c)/m,"fine":float(c)/f},optimizer= optimizer ,metrics=['accuracy'])
            fine_model.load_weights('weights/model_multi_outputmiddle{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f),by_name = True)
            fine_model.fit(x_trains[2], y_trains[2],
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      verbose=1,
                      callbacks = callbacks,
                      validation_data=(x_vals[2], y_vals[2]))
            fine_model.save_weights('weights/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f))

        else :
            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
            TensorBoard(log_dir = logdir4)]
            fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine":
                                         'categorical_crossentropy'},
                loss_weights = {"coarse":1.,"middle":float(c)/m,"fine":float(c)/f},optimizer= optimizer ,metrics=['accuracy'])
            fine_model.load_weights('weights/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f),by_name = True)
            fine_model.fit(x_trains[2], y_trains[2],
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      verbose=1,
                      callbacks = callbacks,
                      validation_data=(x_vals[2], y_vals[2]))
            fine_classif.save_weights('weights/model_multi_outputfine_tuned{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f))
    else:
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,patience =5),
        TensorBoard(log_dir = logdir5)]
        single_model.compile(loss = {"fine":'categorical_crossentropy'},loss_weights = {"fine":float(c)/f},optimizer= optimizer
                           ,metrics=['accuracy'])
        single_model.fit(x_trains[2], y_trains[2]['fine'],
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=1,
                  callbacks = callbacks,
                  validation_data=(x_vals[2], y_vals[2]['fine']))
        single_model.save_weights('weights/model_single{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c, m,f))

else :
    if not os.path.exists('results_{}_id{}.csv'.format(args.dataset,id)):
        df = pd.DataFrame(columns = ['type','perturbation', 'coarse_acc','middle_acc','fine_acc','coarse_conf','middle_conf','fine_conf' ])
        df.to_csv('results_{}_id{}.csv'.format(args.dataset,id), index=False)
    df = pd.read_csv('results_{}_id{}.csv'.format(args.dataset,id))
    per_name = ''
    if args.perturbation=='warp':
        per_name = "warp_s{}_t{}".format(args.s,args.t)
    else :
        per_name = args.perturbation
    y_test_coarse = y_tests['coarse']
    y_test_middle = y_tests['middle']
    y_test_fine = y_tests['fine']
    if not(args.single) :
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
        fine_model.compile(loss = {"coarse": 'categorical_crossentropy',"middle": 'categorical_crossentropy',"fine": 'categorical_crossentropy' },
        loss_weights = {"coarse":1.,"middle":c/m,"fine":c/f},optimizer= optimizer ,metrics=['accuracy'])
        fine_classif.load_weights('weights/model_multi_outputfine{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c,m,f),by_name = True)
        y_pred = fine_classif.predict(x_tests)
        #
        y_coarse = y_pred[0][:,:]
        y_middle = y_pred[1][:,:]
        y_fine = y_pred[2][:,:]

        n = y_test_fine.shape[0]
        y_testfine1 = [np.argmax(y_test_fine[i,:]) for i in range(n)]
        y_test_coarse1 = [np.argmax(y_test_coarse[i,:]) for i in range(n)]
        y_test_middle1 = [np.argmax(y_test_middle[i,:]) for i in range(n)]

        mean = 0.
        fine_conf = 0.
        middle_conf = 0.
        coarse_conf = 0.
        for i in range(x_tests.shape[0]):
            fine_conf+=np.max(y_fine[i,:])
            middle_conf+=np.max(y_middle[i,:])
            coarse_conf+=np.max(y_coarse[i,:])
        fine_conf  *= (1./x_tests.shape[0])
        middle_conf  *= (1./x_tests.shape[0])
        coarse_conf  *= (1./x_tests.shape[0])
        print(fine_conf)
        print(middle_conf)
        print(coarse_conf)
        acc_c,acc_m_f = acc_cmf(y_coarse, y_middle, y_fine,y_testfine1 , args.dataset)
        print(acc_c)
        print(acc_m)
        print(acc_f)
        df = df.append({'type':'multi-output','perturbation':per_name, 'coarse_acc':acc_c,'middle_acc':acc_m,'fine_acc':acc_f, 'coarse_conf':coarse_conf ,'middle_conf': middle_conf,'fine_conf':fine_conf }, ignore_index=True)
        df.to_csv('results_{}_id{}.csv'.format(args.dataset,id), index=False)
    if args.single:
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
        single_model.compile(loss = {"fine":'categorical_crossentropy' },loss_weights = {"fine":3.3},optimizer= optimizer ,metrics=
                           ['accuracy'])
        single_model.load_weights('weights/model_single{}_id{}c{}m{}f{}.h5'.format(args.dataset,id,c,m,f),by_name = True)
        y_pred = single_model.predict(x_tests)
        n = y_pred.shape[0]
        acc_c,acc_m,acc_f = acc_cmf_single(y_pred,y_test_fine,args.dataset)
        (conf_c,conf_m, conf_f) = conf_cmf_single(y_pred,args.dataset)
        print(acc_c)
        print(acc_m)
        print(acc_f)
        print(conf_c)
        print(conf_m)
        print(conf_f)
        df = df.append({'type':'single_output','perturbation':per_name, 'coarse_acc':acc_c,'middle_acc':acc_m,'fine_acc':acc_f}, ignore_index=True)
        df.to_csv('results_MNIST_id{}.csv'.format(id), index=False)
