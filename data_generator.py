""" Code for loading data. """
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import pickle
from utils import get_images
import ipdb

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == '2D':
            self.dim_input = 2
            self.dim_output = 1
            self.input_range = config.get('input_range', [-5.0, 5.0])

        elif FLAGS.datasource == 'plainmulti':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.plainmulti = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.plainmulti:
                metatrain_folders.append(
                    [os.path.join('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(os.path.join('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        elif FLAGS.datasource == 'artmulti':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.artmulti = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi', 'CUB_Bird_blur',
                                'DTD_Texture_blur', 'FGVC_Aircraft_blur', 'FGVCx_Fungi_blur', 'CUB_Bird_pencil',
                                'DTD_Texture_pencil', 'FGVC_Aircraft_pencil', 'FGVCx_Fungi_pencil']
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.artmulti:
                metatrain_folders.append(
                    [os.path.join('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(
                         os.path.join('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        else:
            raise ValueError('Unrecognized data source')

    def make_data_tensor_plainmulti(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task
        # make list of files
        print('Generating filenames')
        all_filenames = []
        for image_itr in range(num_total_batches):
            sel = np.random.randint(4)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            sampled_character_folders = random.sample(folders[sel], self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def make_data_tensor_artmulti(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sel = np.random.randint(12)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            sampled_character_folders = random.sample(folders[sel], self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        # image_file=datadict[filename_queue]
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            # image = tf.convert_to_tensor(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_2D_batch(self, train=False):
        dim_input = self.dim_input
        dim_output = self.dim_output
        batch_size = self.batch_size
        num_samples_per_class = self.num_samples_per_class

        # sin
        amp = np.random.uniform(0.1, 5.0, size=self.batch_size)
        phase = np.random.uniform(0., 2 * np.pi, size=batch_size)
        freq = np.random.uniform(0.8, 1.2, size=batch_size)

        # linear
        A = np.random.uniform(-3.0, 3.0, size=batch_size)
        b = np.random.uniform(-3.0, 3.0, size=batch_size)

        # quadratic
        A_q = np.random.uniform(-0.2, 0.2, size=batch_size)
        b_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        c_q = np.random.uniform(-3.0, 3.0, size=batch_size)

        # cubic
        A_c = np.random.uniform(-0.1, 0.1, size=batch_size)
        b_c = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_c = np.random.uniform(-2.0, 2.0, size=batch_size)
        d_c = np.random.uniform(-3.0, 3.0, size=batch_size)

        # 3d curve
        A_3cur = np.random.uniform(-1.0, 1.0, size=batch_size)
        B_3cur = np.random.uniform(-1.0, 1.0, size=batch_size)

        # ripple
        A_r = np.random.uniform(-0.2, 0.2, size=batch_size)
        B_r = np.random.uniform(-3.0, 3.0, size=batch_size)

        sel_set = np.zeros(batch_size)

        init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
        outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], size=(num_samples_per_class, dim_input))
            sel = np.random.randint(FLAGS.sync_group_num)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            if sel == 0:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func, :, 0:1] + phase[func])
            elif sel == 1:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A[func] * init_inputs[func, :, 0:1] + b[func]
            elif sel == 2:
                # outputs[func] = A_q[func] * np.square(init_inputs[func] - c_q[func]) + b_q[func]
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A_q[func] * np.square(init_inputs[func, :, 0:1]) + b_q[func] * init_inputs[func, :, 0:1] + \
                                c_q[func]
            elif sel == 3:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A_c[func] * np.power(init_inputs[func, :, 0:1],
                                                     np.tile([3], init_inputs[func, :, 0:1].shape)) + b_c[
                                    func] * np.square(init_inputs[func, :, 0:1]) + c_c[func] * init_inputs[func, :, 0:1] + \
                                d_c[func]
            elif sel == 4:
                outputs[func] = A_3cur[func] * np.square(init_inputs[func, :, 0:1]) + B_3cur[func] * np.square(
                    init_inputs[func, :, 1:2])
            elif sel == 5:
                outputs[func] = np.sin(
                    -A_r[func] * (np.square(init_inputs[func, :, 0:1]) + np.square(init_inputs[func, :, 1:2]))) + B_r[func]
            outputs[func] += np.random.normal(0, 0.3, size=(num_samples_per_class, dim_output))
            sel_set[func] = sel
        funcs_params = {'amp': amp, 'phase': phase, 'freq': freq, 'A': A, 'b': b, 'A_q': A_q, 'c_q': c_q, 'b_q': b_q,
                        'A_c': A_c, 'b_c': b_c, 'c_c': c_c, 'd_c': d_c, 'A_3cur': A_3cur, 'B_3cur': B_3cur, 'A_r':A_r, 'B_r':B_r}
        return init_inputs, outputs, funcs_params, sel_set