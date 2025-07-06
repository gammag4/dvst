from src.latent_aggregators import regular_latent_aggregator


class DVSTConfig:
    def __init__(self, N_enc=2, N_dec=7, n_heads=12, d_lat=128, e_ff=2, n_lat=1024, p=16, n_oct=6, C=3, latent_aggregator=regular_latent_aggregator):
        assert d_lat % n_heads == 0, "n_heads should divide d_lat"

        # frames_per_scene is the size of the batches that the videos will be broken into to create scenes
        #   TODO this was from the old idea of breaking scene into smaller batches of 3
        #   TODO test 3 ideas separate:
        #     just incremental creation of latents (testing both w or w/o residual)
        #     just breaking into blocks of 3
        #     and test both together (breaking into blocks of size k and doing incremental creation of latents)
        # self.frames_per_scene = frames_per_scene
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.n_heads = n_heads
        self.d_lat = d_lat
        self.e_ff = e_ff
        self.n_lat = n_lat
        self.p = p
        self.n_oct = n_oct
        self.C = C
        self.latent_aggregator = latent_aggregator
