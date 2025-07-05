class DVSTConfig:
    def __init__(self, frames_per_scene=3, N_enc=2, N_dec=7, n_heads=12, d_lat=128, e_ff=2, n_lat=1024, p=16, n_oct=6, C=3):
        assert d_lat % n_heads == 0, "n_heads should divide d_lat"

        self.frames_per_scene = frames_per_scene
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.n_heads = n_heads
        self.d_lat = d_lat
        self.e_ff = e_ff
        self.n_lat = n_lat
        self.p = p
        self.n_oct = n_oct
        self.C = C
