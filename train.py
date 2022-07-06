from __future__ import absolute_import, division, print_function, unicode_literals
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
        w = np.ones((n_train,1), dtype='float32')
        
        # 10d of Gaussian
        # x = BR_data.gen_nd_Gaussian_w_hole(10, n_train, 1.0)
        # np.savetxt('./dataset_for_training/Gaussian_10d_mixture.dat', x)
        x = np.loadtxt('./dataset_for_training/Gaussian_10d_mixture.dat').astype(np.float32)[:n_train,:]
        return x,w

    #differential entropy of the 2d Gaussian mixture distribution
    #ce = 4.59343477271186

    x,w = get_data(args.n_train, args.n_dim)


    data_flow = BR_data.dataflow(x, buffersize=args.n_train, batchsize=args.batch_size, y=w)
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
            layers.Dense(256, activation=None, input_shape=input_shape),
            layers.LeakyReLU(0.2),
            layers.Dense(256, activation=None),
            layers.LeakyReLU(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

    # define generator
    def make_generator(input_shape):
        return tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(10)
        ])

    # call model one to complete the building process
    x_init = data_flow.get_n_batch_from_shuffled_batched_dataset(1)
    kr_model = create_kr_model()
    kr_model(x_init)
    Generator = make_generator(args.n_dim)
    Discriminator = make_discriminator(args.n_dim + 1)

    # define loss function
    def get_kr_loss(x, w):
        pdf = kr_model(x)
        loss = -tf.reduce_mean(pdf*w)

        reg = tf.constant(0.0)

        loss += reg

        return loss, reg


    def get_d_loss(real, fake):
        return -tf.reduce_mean(tf.math.log(real + 1e-10) + tf.math.log(1. - fake + 1e-10))

    def get_g_loss(fake):
        return -tf.reduce_mean(tf.math.log(fake + 1e-10))

    def inv_sphere_proj(raw_point, n_dim, n_train, radius):
        """inverse stereographic projection"""
        res = np.zeros((n_train, n_dim + 1), dtype='float32')
        s_sq = np.linalg.norm(raw_point, axis=1) * np.linalg.norm(raw_point, axis=1)
        for i in range(n_train):
            for j in range(n_dim):
                res[i][j] = (2 * raw_point[i][j] / (s_sq[i] + 1) * radius).sum()
            res[i][-1] = ((s_sq[i] - 1) / (s_sq[i] + 1) * radius).sum()
        return res


    # metrics for loss and regularization
    loss_metric = tf.keras.metrics.Mean()   #加权平均
    reg_metric = tf.keras.metrics.Mean()

    # stochastic gradient method ADAM
    kr_optim = tf.keras.optimizers.Adam(learning_rate=args.kr_lr)
    g_optim = tf.keras.optimizers.Adam(learning_rate=args.g_lr)
    d_optim = tf.keras.optimizers.Adam(learning_rate=args.d_lr)

    # check point and initialization
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=kr_optim, net=kr_model)
    manager=tf.train.CheckpointManager(ckpt, args.ckpts_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print(" ------ Initializing from scratch ------")
        m = 1 # the number minibatches used for data initialization
        x_init = data_flow.get_n_batch_from_shuffled_batched_dataset(m)
        #if args.rotation:
        #    pdf_model.WLU_data_initialization()
        kr_model.actnorm_data_initialization()
        kr_model(x_init)
    # summary
    summary_writer = tf.summary.create_file_writer(args.summary_dir)

    # prepare one training iteration step
    @tf.function
    def kr_train_step(inputs, vars):
        x_t, w_t = inputs
        with tf.GradientTape() as kr_tape:
            x_t = kr_model(x_t)
            x_t = inv_sphere_proj(x_t, args.n_dim, x_t.shape()[0], args.radius)
            sample_gaussian_vector = BR_data.gen_nd_Gaussian(args.n_dim + 1, x_t.shape()[0])
            y_t = x_t + sample_gaussian_vector

            g_l, d_l = 0.0, 0.0
            for i in range(args.g_epoch):
                g_l, d_l = gan_train_step(y_t)

            loss, reg = get_kr_loss(y_t, w_t)

        grads = kr_tape.gradient(loss, vars)
        kr_optim.apply_gradients(zip(grads, vars))

        return loss, g_l, d_l, reg


    def gan_train_step(real_inputs):
        z = BR_data.gen_nd_Gaussian(real_inputs.shape()[1], real_inputs.shape()[0])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_input = Generator(z, training = True)
            fake_input = inv_sphere_proj(fake_input, args.n_dim, fake_input.shape()[0], args.radius)
            sample_gaussian_vector = BR_data.gen_nd_Gaussian(args.n_dim + 1, fake_input.shape()[0])
            fake_input = fake_input + sample_gaussian_vector


            for i in range(args.d_epoch):
                fake_ans = Discriminator(fake_input, training = True)
                real_ans = Discriminator(real_inputs, training = True)
                d_loss = get_d_loss(real_ans, fake_ans)
                d_grad = d_tape.gradient(d_loss, Discriminator.trainable_variables)
                d_optim.apply_gradients(zip(d_grad, Generator.trainable_variables))

        g_loss = get_g_loss(fake_input)
        g_grad = g_tape.gradient(g_loss, Generator.trainable_variables)
        g_optim.apply_gradients(zip(g_grad, Generator.trainable_variables))
        return g_loss, d_loss

    # used for the computation of KL divergence
    # n_valid = 320000
    # y = np.loadtxt('./dataset_for_training/Logistic_8d_w_2d_holes_valid.dat').astype(np.float32)[:n_valid,:]
    # y, _ = BR_data.gen_xd_Logistic_w_2d_hole(args.n_dim, n_valid, 2.0, 3.0, np.pi/4.0, 7.6)

    loss_vs_epoch=[]
    KL_vs_epoch=[]

    n_epochs = args.n_epochs
    with summary_writer.as_default():
        # iterate over epochs
        iteration = 0
        for i in range(1,n_epochs+1):
            # freeze the rotation layers after a certain number of epochs
            g_vars = kr_model.trainable_weights

            #if args.rotation == True and i >= args.rotation_epochs:
            #    g_vars = [var for var in g_vars if not 'rotation' in var.name]

            start_time = time.time()
            # iterate over the batches of the dataset
            for step, train_batch in enumerate(train_dataset):
                loss, reg = kr_train_step((train_batch[0], train_batch[1]), g_vars)

                loss_metric(loss-reg)

                iteration += 1

                # write the summary file
                #if tf.equal(optimizer.iterations % args.log_step, 0):
                #    tf.summary.scalar('loss', loss_metric.result(), step=optimizer.iterations)

            # difference between the approximated cross entropy and the entropy
            kl_d = loss_metric.result() - 4.59343477271186


            print('epoch %s, iteration %s, loss = %s,  kl_d = %s, time = %s' %
                             (i, iteration, loss_metric.result().numpy(), kl_d.numpy(), time.time()-start_time))

            loss_vs_epoch += [loss_metric.result().numpy()]
            KL_vs_epoch += [kl_d.numpy()]

            loss_metric.reset_states()

            # re-shuffle the dataset
            train_dataset = data_flow.update_shuffled_batched_dataset()

            ckpt.step.assign_add(1)
            if int(ckpt.step) % args.ckpt_step == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if i % args.n_draw_samples == 0:
                xs = kr_model.draw_samples_from_prior(args.n_samples, args.n_dim)
                ys = kr_model.mapping_from_prior(xs)
                np.savetxt('epoch_{}_prior.dat'.format(i), xs.numpy())
                np.savetxt('epoch_{}_sample.dat'.format(i), ys.numpy())

        c1=np.array(range(1,n_epochs+1)).reshape(-1,1)
        c2=np.array(loss_vs_epoch).reshape(-1,1)
        c3=np.array(KL_vs_epoch).reshape(-1,1)
        np.savetxt('cong_vs_epoch.dat',np.concatenate((c1, c2, c3), axis=1))

if __name__ == '__main__':
    from configargparse import ArgParser
    p = ArgParser()
    # Data arguments
    p.add('--data_dir', type=str, help='Path to preprocessed data files.')

    # save parameters
    p.add('--ckpts_dir', type=str, default='./pdf_ckpt', help='Path to the check points.')
    p.add('--summary_dir', type=str, default='./kr_summary', help='Path to the summaries.')
    p.add('--log_step', type=int, default=16, help='Record information every n optimization iterations.')
    p.add('--ckpt_step', type=int, default=50, help='Save the model every n epochs.')

    # Neural network hyperparameteris
    p.add('--n_depth', type=int, default=8, help='The number of affine coupling layers.')
    p.add('--n_width', type=int, default=32, help='The number of neurons for the hidden layers.')
    p.add('--n_step', type=int, default=1, help='The step size for dimension reduction in each squeezing layer.')
    p.add('--rotation', action='store_true', help='Specify rotation layers or not?')
    p.add('--d_epoch', type=int, default=1, help='The number discriminator updates during one epoch of generator')
    p.add('--g_epoch', type=int, default=1, help='The number generator updates during one epoch of krnet')
    #p.set_defaults(rotation=True)
    p.add('--n_bins4cdf', type=int, default=0, help='The number of bins for uniform partition of the support of PDF.')
    p.add('--flow_coupling', type=int, default=1, help='Coupling type: 0=additive, 1=affine.')
    p.add('--h1_reg', action='store_true', help='H1 regularization of the PDF.')
    p.add('--shrink_rate', type=float, default=0.9, help='The shrinking rate of the width of NN.')

    #optimization hyperparams:
    p.add("--n_dim", type=int, default=10, help='The number of random dimension.')
    p.add("--n_train", type=int, default=10000, help='The number of samples in the training set.')
    p.add('--batch_size', type=int, default=10000, help='Batch size of training generator.')
    p.add("--kr_lr", type=float, default=0.001, help='Base krnet learning rate.')
    p.add("--g_lr", type=float, default=0.0004, help='Base generator learning rate.')
    p.add("--d_lr", type=float, default=0.0004, help='Base discriminator learning rate.')
    p.add('--n_epochs',type=int, default=6000, help='Total number of training epochs.')

    # samples:
    p.add("--n_samples", type=int, default=10000, help='Sample size for the trained model.')
    p.add("--n_draw_samples", type=int, default=1000, help='Draw samples every n epochs.')

    p.add("--radius", type=float, default=1.0, help='The radius of the sphere')

    args = p.parse_args()
    main(args)
