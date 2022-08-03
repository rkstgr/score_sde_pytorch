import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 2
    training.n_iters = 1300001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 1
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 64
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'MTG'
    data.dataset_path = "/Volumes/Black T5/dataset/mtg-jamendo/mtg_jamendo.py"
    data.cache_dir = "/Volumes/Black T5/dataset/huggingface_cache"
    data.normalizers_path = "/Users/erik/Dev/projects/score_sde/audio/audio_normalizers.pckl"
    data.num_proc = 4
    data.n_fft = 1024
    data.hop_length = 431
    data.duration = 10
    data.sampling_rate = 22050
    data.genre = 'lofi'
    data.uniform_dequantization = False
    data.processing_batch_size = 8
    data.image_size = 512
    data.num_channels = 2

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config