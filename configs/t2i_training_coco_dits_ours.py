import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

text_model=Args(
    # patch 2 (default)
    hidden_size = 512, # mlp -> 512 * 4
    num_heads = 8, 
    depth = 17,
    latent_size = 32,
    learn_sigma = False, # different from DiT, we direct predict noise here
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=False, # for larger model
    cfg_indicator=0.10,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
)
img_model=Args(
    # patch 2 (default)
    hidden_size = 512, # mlp -> 512 * 4
    num_heads = 8, 
    depth = 17,
    latent_size = 32,
    learn_sigma = False, # different from DiT, we direct predict noise here
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=False, # for larger model
    cfg_indicator=0.10,
    textVAE = None
)



def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234                                          # random seed
    config.z_shape = (4, 32, 32)                                # image latent size

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth', # path of pretrained VAE CKPT from LDM
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,
        batch_size=1024,
        mode='cond',
        log_interval=10,
        eval_interval=1000,
        save_interval=5000,
        n_samples_eval=5,  
    )

    config.optimizer = d(
        name='adamw',
        lr=0.00001,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    global text_model
    config.text_nnet = d(
        name='dit',
        model_args=text_model,
    )
    global img_model
    config.img_nnet = d(
        name='dit',
        model_args=img_model,
    )
    config.loss_coeffs = [1/4, 1]
                           # weight on loss, only needed for DiMR. Here, loss = 1/4 * loss_block1 + 1/2 * loss_block2 + 1 * loss_block3
    
    config.dataset = d(
        name='JDB_demo_features',                               # dataset name
        resolution=256,                                         # dataset resolution (512)
        llm='clip',                                             # language model to generate language embedding
        train_path='COCO_train',     # training set path
        val_path='./',      # val set path
        cfg=False
    )

    config.sample = d(
        sample_steps=50,                                        # sample steps duing inference/testing
        n_samples=30000,                                        # number of samples for testing (during training, we sample 10K images, which is hardcoded in the training script)
        mini_batch_size=10,                                     # batch size for testing (i.e., the number of images generated per GPU)
        cfg=False,
        scale=7,                                                # cfg scale
        path=''
    )
    
    config.exp_name = 'dimrxl_2r_ours'  # experiment name

    return config
