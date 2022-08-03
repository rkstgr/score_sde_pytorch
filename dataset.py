# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import tensorflow as tf
import tensorflow_datasets as tfds
import torch


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'SVHN':
        dataset_builder = tfds.builder('svhn_cropped')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'CELEBA':
        dataset_builder = tfds.builder('celeb_a')
        train_split_name = 'train'
        eval_split_name = 'validation'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == 'LSUN':
        dataset_builder = tfds.builder(f'lsun/{config.data.category}')
        train_split_name = 'train'
        eval_split_name = 'validation'

        if config.data.image_size == 128:
            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:
            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = eval_split_name = 'train'


    elif config.data.dataset == "MTG":

        normalizers = load_normalizers(config.data.normalizers_path)

        def prepare_dataset(ds: datasets.Dataset, batch_size: int) -> datasets.Dataset:

            sr = config.data.sampling_rate
            n_fft = config.data.n_fft
            hop_length = config.data.hop_length
            duration = config.data.duration
            time_bins = int(np.ceil(sr * duration / hop_length))

            map_params = dict(
                batched=True,
                batch_size=config.data.processing_batch_size,
                num_proc=config.data.num_proc
            )

            def preprocess_fn(d):
                """Basic preprocessing function scales data to [0, 1) and randomly flips."""
                img = d['image']
                if uniform_dequantization:
                    img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

                return dict(image=img, label=d.get('label', None))

            hf_ds = (ds
                     .filter(partial(filter_func, genre=config.data.genre), **map_params)
                     .cast_column("audio", datasets.Audio(sampling_rate=sr, decode=False))
                     .map(partial(load_audio, sampling_rate=sr, remove_silence=True, duration=duration),
                          **map_params)
                     .map(partial(create_spectrogram, n_fft=n_fft, hop_length=hop_length),
                          **map_params)
                     .map(crop_spectrogram,
                          **map_params)
                     .map(partial(normalize_spectrogram, normalizers=normalizers),
                          **map_params)
                     .cast_column("audio_spectrogram",
                                  datasets.Array3D((n_fft // 2, time_bins, config.data.num_channels), "float32"))
                     .rename_column("audio_spectrogram", "image")
                     )
            ds = hf_ds.to_tf_dataset(batch_size=per_device_batch_size, columns=["id", "image"],
                                     drop_remainder=True)
            ds = ds.repeat(count=num_epochs)
            ds = ds.shuffle(shuffle_buffer_size)
            ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for bs in reversed(batch_dims[:-1]):
                ds = ds.batch(bs, drop_remainder=True)
            return ds.prefetch(prefetch_size)

        train_hf = datasets.load_dataset(path=config.data.dataset_path,
                                         cache_dir=config.data.cache_dir,
                                         split=datasets.Split.TRAIN,
                                         ignore_verifications=True)

        valid_hf = datasets.load_dataset(path=config.data.dataset_path,
                                         cache_dir=config.data.cache_dir,
                                         split=datasets.Split.VALIDATION,
                                         ignore_verifications=True)

        train_ds = prepare_dataset(train_hf, batch_size)
        valid_ds = prepare_dataset(valid_hf, batch_size)

        return train_ds, valid_ds, None


    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ['FFHQ', 'CelebAHQ']:
        def preprocess_fn(d):
            sample = tf.io.parse_single_example(d, features={
                'shape': tf.io.FixedLenFeature([3], tf.int64),
                'data': tf.io.FixedLenFeature([], tf.string)})
            data = tf.io.decode_raw(sample['data'], tf.uint8)
            data = tf.reshape(data, sample['shape'])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
            return dict(image=img, label=None)

    else:
        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d['image'])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

            return dict(image=img, label=d.get('label', None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=True, read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder
