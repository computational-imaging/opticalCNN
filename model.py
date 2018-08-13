import os

import tensorflow as tf
import abc

class Model(abc.ABC):
    """Generic tensorflow model class.
    """
    def __init__(self, name, ckpt_path=None):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=sess_config)
        self.name = name
        self.ckpt_path = ckpt_path

    @abc.abstractmethod
    def _build_graph(self, x_train, **kwargs):
        """Builds the model, given x_train as input.

        Args:
            x_train: The dequeued training example
            **kwargs: Model parameters that can later be passed to the "fit" function

        Returns:
            model_output: The output of the model
        """

    @abc.abstractmethod
    def _get_data_loss(self,
                      model_output,
                      ground_truth):
        """Computes the data loss (not regularization loss) of the model.

        !!For consistency of weighing of regularization loss vs. data loss,
        normalize loss by batch size!!

        Args:
            model_output: Output of self._build_graph
            ground_truth: respective ground truth

        Returns:
            data_loss: Scalar data loss of the model.         """

    def _get_reg_loss(self):
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return reg_loss


    def infer(self, model_inputs, model_params={}, get_loss=False, gt=None):
        """Does inference at test time.
        """
        x_test, y_test = self._get_inference_queue()

        # Set up the training graph
        with tf.variable_scope('model'):
            model_output_graph = self._build_graph(x_test, **model_params)

            if get_loss:
                data_loss_graph = self._get_data_loss(model_output_graph, y_test)

        # Create a saver
        self.saver = tf.train.Saver()

        if self.ckpt_path is not None:
            self.saver.restore(self.sess,self.ckpt_path)
        else:
            print("Warning: No checkpoint path given. Inference happens with random weights")

        # Init op
        init = tf.global_variables_initializer()
        self.sess.run(init)

        print("Starting Queues")
        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        model_outputs = []
        try:
            while True:
                model_output= self.sess.run(model_output_grpah)
                model_outputs.append(model_output)

                if coord.should_stop():
                   break

        except Exception as e:
            print("Interrupted due to exception")
            print(e)
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)

        return model_outputs


    @abc.abstractmethod
    def _get_training_queue(self, batch_size):
        """Builds the queues for training data.

        Use tensorflow's readers, decoders and tf.train.batch to build the dataset.

        Args:
            batch_size:

        Returns:
            x_train: the dequeued model input
            y_train: the dequeued ground truth

        Sketch of minimum example:
            def _get_training_queue(self, batch_size):
                file_list = tf.matching_files('./test_imgs/*.png')
                filename_queue = tf.train.string_input_producer(file_list)

                image_reader = tf.WholeFileReader()
                _, image_file = image_reader.read(filename_queue)
                image = tf.image.decode_png(image_file,
                                            channels=1,
                                            dtype=tf.uint8)
                image = tf.cast(image, tf.float32)
                image /= 255.0

                image_batch = tf.train.batch(image,
                                             shapes=[512,512,1],
                                             batch_size=batch_size)
                return image_batch
        """


    def _get_validation_queue(self):
        """

        Returns:

        """

    def fit(self,
            model_params, # Dictionary of model parameters
            opt_type, # Type of optimization algorithm
            opt_params, # Parameters of optimization algorithm
            batch_size,
            starter_learning_rate,
            adadelta_learning_rate,
            logdir,
            num_steps,
            num_steps_until_save,
            num_steps_until_summary,
            decay_type=None, # Type of decay
            decay_params=None, # Decay parameters
            ):
        """Trains the model.
        """
        x_train, y_train = self._get_training_queue(batch_size)

        print("\n\n")
        print(40*"*")
        print("Saving model and summaries to %s"%logdir)
        print("Optimization parameters:")
        print(opt_type)
        print(opt_params)
        print("Starter learning rate is %f"%starter_learning_rate)
        print(40*"*")
        print("\n\n")

        # Set up the training graph
        with tf.variable_scope('model'):
            model_output_train = self._build_graph(x_train, **model_params)
            data_loss_graph = self._get_data_loss(model_output_train, y_train)
            reg_loss_graph = self._get_reg_loss()
            total_loss_graph = tf.add(reg_loss_graph,
                                      data_loss_graph)

        if decay_type is not None:
            global_step = tf.Variable(0, trainable=False)

            if decay_type == 'exponential':
                learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
            elif decay_type == 'polynomial':
                learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
        else:
            learning_rate = starter_learning_rate

        if opt_type == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #,
                                               #**opt_params)
        elif opt_type == 'sgd_with_momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               **opt_params)
        elif opt_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=adadelta_learning_rate, rho=.9)
                                               #**opt_params)

        if decay_type is not None:
            train_step = optimizer.minimize(total_loss_graph, global_step=global_step)
        else:
            train_step = optimizer.minimize(total_loss_graph)

        # Attach summaries to some of the training parameters
        tf.summary.scalar('data_loss', data_loss_graph)
        tf.summary.scalar('reg_loss', reg_loss_graph)
        tf.summary.scalar('total_loss', total_loss_graph)
        tf.summary.scalar('learning_rate', learning_rate)

        # Create a saver
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2,
                                    max_to_keep=3)

        # Get all summaries
        summaries_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir, self.sess.graph, flush_secs=60)

        if self.ckpt_path is not None:
            self.saver.restore(self.sess,self.ckpt_path)

        # Init op
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Train the model
        print("Starting Queues")
        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        print("Beginning the training")
        try:
            for step in range(num_steps):
                _, total_loss, reg_loss, data_loss = self.sess.run([train_step,
                                                                    total_loss_graph,
                                                                    reg_loss_graph,
                                                                    data_loss_graph])
                if not step % 20:
                    print("Step %d\n    total_loss %0.8f   reg_loss %0.8f   data_loss %0.8f\n"%\
                        (step, total_loss, reg_loss, data_loss))

                if coord.should_stop():
                   break

                if not step % num_steps_until_save and step:
                    print("Saving model...")
                    save_path = os.path.join(logdir, self.name+'.ckpt')
                    self.saver.save(self.sess, save_path, global_step=step)

                if not step % num_steps_until_summary:
                    print("Writing summaries...")
                    summary = self.sess.run(summaries_merged)
                    summary_writer.add_summary(summary, step)
        except Exception as e:
            print("Training interrupted due to exception")
            print(e)
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)
