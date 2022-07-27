from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import sys
sys.path.append('./')
import BR_lib.BR_model as BR_model
import BR_lib.BR_data as BR_data


# choose GPU 0 or 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



def main(args):
    def get_data(n_train, n_dim):
        # 10d of Gaussian
        # x = BR_data.gen_nd_Gaussian(n_dim, n_train)
        # np.savetxt('./dataset_for_training/Gaussian_10d_mixture.dat', x)
        x = np.loadtxt('./dataset_for_training/Gaussian_10d_mixture.dat').astype(np.float32)[:n_train,:]
        return x
    print("------ Loading data ------")
    x = get_data(args.n_train, args.n_dim)


    data_flow = BR_data.dataflow(x, buffersize=args.n_train, batchsize=args.batch_size)
    train_dataset = data_flow.get_shuffled_batched_dataset()

    # create the model
    def create_kr_model():
        # build up the model
        pdf_model = BR_model.IM_rNVP_KR_CDF('pdf_model_KR_CDF',
                                            args.n_dim,
                                            args.n_step,
                                            args.n_depth,
                                            n_width=args.n_width,
                                            n_bins=args.n_bins4cdf,
                                            shrink_rate=args.shrink_rate,
                                            flow_coupling=args.flow_coupling,
                                            rotation=args.rotation)
        return pdf_model

    # define discriminator
    def make_discriminator(input_shape):
        return tf.keras.Sequential([
            layers.Dense(256, activation=None, input_shape=input_shape, ),
            layers.LeakyReLU(0.8),
            layers.Dense(64, activation=None),
            layers.LeakyReLU(0.8),
            layers.Dense(1, activation='sigmoid')
        ])

    # define generator
    def make_generator(input_shape):
        return tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(10)
        ])

    # call model one to complete the building process
    print("------ Initializing data ------")
    x_init = data_flow.get_n_batch_from_shuffled_batched_dataset(1)
    kr_model = create_kr_model()
    kr_model(x_init)
    Generator = make_generator((args.n_dim,))
    Discriminator = make_discriminator((args.n_dim + 1,))

    #trainable parameters
    kr_para = kr_model.trainable_variables
    g_para = Generator.trainable_variables
    d_para = Discriminator.trainable_variables

    # define loss function
    def get_kr_loss(x):
        y = kr_model(x)[0]
        y = inv_sphere_proj(y, args.n_dim, y.shape[0], args.radius)
        y = Discriminator(y)
        w = tf.ones((x.shape[0],1), tf.float32)
        loss = -tf.reduce_mean(y * w)

        reg = tf.constant(0.0)

        loss += reg

        return loss, reg


    def get_d_loss(real, fake):
        # return -tf.reduce_mean(real) - tf.reduce_mean(1. - fake)
        return -tf.reduce_mean(tf.math.log(real + 1e-10) + tf.math.log(1. - fake + 1e-10))

    def get_g_loss(fake):
        # return tf.reduce_mean(1. - fake)
        return -tf.reduce_mean(tf.math.log(fake + 1e-10))

    def inv_sphere_proj(raw_point, n_dim, n_train, radius):
        """inverse stereographic projection

        n_dim:
        the dimension of the original data

        n_train:
        the number of points to be projected

        radius:
        the radius of the sphere
        """

        res = []
        for i in range(n_train):
            tmp = []
            normal = tf.sqrt(tf.reduce_sum(tf.square(raw_point[i])))
            for j in range(n_dim):
                tmp.append(tf.reduce_sum((2 * raw_point[i][j] / (normal + 1) * radius)))
            tmp.append(tf.reduce_sum(((normal - 1) / (normal + 1) * radius)))
            res.append(tf.stack(tmp))
        res = tf.stack(res)
        return res

    # stochastic gradient method ADAM
    kr_optim = tf.keras.optimizers.Adam(learning_rate=args.kr_lr)
    g_optim = tf.keras.optimizers.Adam(learning_rate=args.g_lr)
    d_optim = tf.keras.optimizers.Adam(learning_rate=args.d_lr)

    def g_metric_base(c, n_dim):
        """
        TV distance 解析解

        c:

        n_dim:
            dimension of the data
        """
        r_sq = n_dim * (1 + c) * np.log(1 + c) / c
        return tf.math.igamma(n_dim / 2, r_sq / 2) - tf.math.igamma(n_dim / 2, r_sq / (1 + c) / 2)

    def d_real_loss(real, fake):
        """
        calculate the expectation of
            real + 1 - fake

        :param real:
        :param fake:
        :return:
        """
        return tf.reduce_mean(real) + tf.reduce_mean(1. - fake)

    def g_data(n_dim, c, n_sample=1000):
        """
        generate two groups of data

        :param n_dim:
            dimension
        :param c:

        :param n_sample:
            number of the data in one group
        :return:
            data group1, data group2
        """
        mean = np.zeros(n_dim)  # mean of the data
        var1 = np.identity(n_dim)   # covariance matrix of data1
        var2 = np.diag([1 + c] * n_dim) # covariance matrix of data2

        np.random.seed(42)  # set random seed
        # generate data randomly
        ori = np.random.multivariate_normal(mean, var1, n_sample)   # data 1
        contrast = np.random.multivariate_normal(mean, var2, n_sample) # data 2

        return ori, contrast

    ori, contrast = g_data(args.n_dim + 1, args.c)

    def g_metric(ori, contrast):
        # get the probabilities
        ori = Discriminator(ori, training=False)
        contrast = Discriminator(contrast, training=False)

        # get D^
        ori = tf.where(ori < 0.5, 0., 1.)
        contrast = tf.where(contrast < 0.5, 0., 1.)

        tf.print(tf.reduce_sum(ori))
        tf.print(tf.reduce_sum(contrast))

        # print calculated TVD
        tf.print(d_real_loss(ori, contrast))
        return

    metric = g_metric_base(args.c, args.n_dim)

    # prepare one training iteration step
    def kr_train_step(inputs, kr_para, g_para, d_para):
        x_t = inputs
        d_loss, g_loss = tf.constant(0.0), tf.constant(0.0)
        with tf.GradientTape() as kr_tape:
            """generate real data"""
            y_t = kr_model(x_t)[0]
            y_t = inv_sphere_proj(y_t, args.n_dim, x_t.shape[0], args.radius)   #invert sphere projection
            sample_gaussian_vector = BR_data.gen_nd_Gaussian(args.n_dim + 1, y_t.shape[0])  #generate stochastic noise
            y_t = y_t + sample_gaussian_vector

            """generator step"""
            for i in tf.range(args.g_epoch):
                tf.print("------gan_step",i + 1,"------")
                real_inputs = y_t
                z = BR_data.gen_nd_Gaussian(args.n_dim, real_inputs.shape[0]) #stochastic input
                with tf.GradientTape() as g_tape:
                    """generate fake data"""
                    fake_inputs = Generator(z, training=True)  #generate fake data
                    fake_inputs = inv_sphere_proj(fake_inputs, args.n_dim, fake_inputs.shape[0], args.radius) #invert sphere projection
                    sample_gaussian_vector = BR_data.gen_nd_Gaussian(args.n_dim + 1, fake_inputs.shape[0]) #generate stochastic noise
                    fake_inputs = fake_inputs + sample_gaussian_vector

                    pre_d_loss = tf.constant(1000.)

                    for j in tf.range(args.d_epoch):
                        with tf.GradientTape() as d_tape:
                            """discriminator step"""
                            # tf.print("------dis_step", j + 1,"------")

                            fake_ans = Discriminator(fake_inputs, training=True)
                            real_ans = Discriminator(real_inputs, training=True)
                            # tf.print("real", tf.reduce_sum(real_ans))
                            # tf.print("fake", tf.reduce_sum(fake_ans))
                            d_loss = get_d_loss(real_ans, fake_ans)     #get the loss of discriminator
                        d_grad = d_tape.gradient(d_loss, d_para)    #calculate the gradient
                        # tf.print("------updating discriminator------")
                        # tf.print("pre_tar", pre_d_loss, "tar", d_loss)
                        d_optim.apply_gradients(zip(d_grad, d_para))
                        if (tf.abs(pre_d_loss - d_loss) < 0.0001):
                            break
                            # continue
                        pre_d_loss = d_loss    #update discriminator parameters
                    g_loss = get_g_loss(fake_ans)    #get the loss of generator
                g_grad = g_tape.gradient(g_loss, g_para)    #calculate the gradient
                tf.print("------updating generator------")
                tf.print("fake", tf.reduce_sum(fake_ans))
                tf.print("g_loss", g_loss)
                g_optim.apply_gradients(zip(g_grad, g_para))    #update generator parameters
                tf.print(metric)
                g_metric(ori, contrast)

            loss, reg = get_kr_loss(x_t)    #get the loss of krnet
        tf.print("------updating krnet------")
        tf.print("kr_loss", loss)
        grads = kr_tape.gradient(loss, kr_para) #calculate the gradient

        kr_optim.apply_gradients(zip(grads, kr_para))   #update krnet parameters
        return loss, g_loss, d_loss


    n_epochs = args.n_epochs
    # iterate over epochs
    print("---------------start-training-----------------")
    for i in tf.range(1,n_epochs+1):
        # iterate over the batches of the dataset
        for step, train_batch in enumerate(train_dataset):
            tf.print("step", step + 1)
            loss, g_l, d_l = kr_train_step(train_batch, kr_para, g_para, d_para)
            # loss, g_l, d_l, reg = D_step(train_batch, kr_para, g_para, d_para)
            tf.print("kr_loss =", loss,"g_l =", g_l, "d_l =", d_l)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Data arguments
    p.add_argument('--data_dir', type=str, help='Path to preprocessed data files.')

    # save parameters
    p.add_argument('--ckpts_dir', type=str, default='./pdf_ckpt', help='Path to the check points.')
    p.add_argument('--summary_dir', type=str, default='./kr_summary', help='Path to the summaries.')
    p.add_argument('--log_step', type=int, default=16, help='Record information every n optimization iterations.')
    p.add_argument('--ckpt_step', type=int, default=50, help='Save the model every n epochs.')

    # Neural network hyperparameteris
    p.add_argument('--n_depth', type=int, default=8, help='The number of affine coupling layers.')
    p.add_argument('--n_width', type=int, default=32, help='The number of neurons for the hidden layers.')
    p.add_argument('--n_step', type=int, default=1, help='The step size for dimension reduction in each squeezing layer.')
    p.add_argument('--rotation', action='store_true', help='Specify rotation layers or not?')
    p.add_argument('--d_epoch', type=int, default=500, help='The number discriminator updates during one epoch of generator')
    p.add_argument('--g_epoch', type=int, default=10, help='The number generator updates during one epoch of krnet')
    #p.set_defaults(rotation=True)
    p.add_argument('--n_bins4cdf', type=int, default=0, help='The number of bins for uniform partition of the support of PDF.')
    p.add_argument('--flow_coupling', type=int, default=1, help='Coupling type: 0=additive, 1=affine.')
    p.add_argument('--h1_reg', action='store_true', help='H1 regularization of the PDF.')
    p.add_argument('--shrink_rate', type=float, default=0.9, help='The shrinking rate of the width of NN.')

    #optimization hyperparams:
    p.add_argument("--n_dim", type=int, default=10, help='The number of random dimension.')
    p.add_argument("--n_train", type=int, default=100, help='The number of samples in the training set.')
    p.add_argument('--batch_size', type=int, default=50, help='Batch size of training generator.')
    p.add_argument("--kr_lr", type=float, default=0.001, help='Base krnet learning rate.')
    p.add_argument("--g_lr", type=float, default=0.0001, help='Base generator learning rate.')
    p.add_argument("--d_lr", type=float, default=0.001, help='Base discriminator learning rate.')
    p.add_argument('--n_epochs',type=int, default=10, help='Total number of training epochs.')

    p.add_argument('--c', type=float, default=0.01, help='Total number of training epochs.')

    # samples:
    p.add_argument("--n_samples", type=int, default=10000, help='Sample size for the trained model.')
    p.add_argument("--n_draw_samples", type=int, default=1000, help='Draw samples every n epochs.')

    p.add_argument("--radius", type=float, default=1.0, help='The radius of the sphere')

    args = p.parse_args()
    main(args)
