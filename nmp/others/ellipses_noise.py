import torch
from addict import Dict

# create dictionary with parameters
cfg = Dict()

cfg.args.n_noisy = 3  # number of noisy images
cfg.noise.n_ellipses = 2
cfg.noise.radius.low = 5
cfg.noise.radius.high = 10
cfg.noise.gaussian_var = 0.25

cfg.ds.res = 32  # image resolution
cfg.ds.n_channels = 1  # number of color channels


# @torch.jit.script
def create_elliptic_mask(size: int, center: torch.Tensor, radius: torch.Tensor,
                         ellip: torch.Tensor):
    """

    :param size: (scalar), e.g. x_res=y_res=32
    :param center: (n_ellipses=4, n_noisy=3, n_dim=2 (xy))
    :param radius: (n_ellipses=4, n_noisy=3)
    :param ellip:  (n_ellipses=4, n_noisy=3), ellip=1 creates a circle
    :return: (n_noisy=3, size=64, size=64])
    """
    x = torch.arange(size, dtype=torch.float32)[:, None]  # (64, 1)
    y = torch.arange(size, dtype=torch.float32)[None]  # (1, 64)

    # distance of each pixel to the ellipsis' center (4, 3, 64, 64)
    dist_from_center = torch.sqrt(
        ellip[:, :, :, None, None] * (x - center[:, :, :, 0:1, None]) ** 2
        + (y - center[:, :, :, 1:2, None]) ** 2 / ellip[:, :, :, None, None])
    # dist_from_center = torch.sqrt(ellip*(x - center[0])**2 + (y - center[1])**2/ellip)

    masks = dist_from_center <= radius[:, :, :, None, None]
    mask, _ = torch.max(masks, dim=1)
    return mask  # (n_noisy=3, size=64, size=64])


# @torch.jit.script
def apply_mask_and_noise(mask: torch.Tensor, noise: torch.Tensor,
                         img: torch.Tensor, n_noisy: int,
                         n_channels: int):  # , translation: torch.Tensor
    imgs = img.repeat(1, n_noisy + 1, 1, 1, 1)

    if n_channels == 3:
        # apply noise and mask on all RGB color channels equally
        noise = noise.repeat(1, 3, 1, 1)
        mask = mask[:, None].repeat(1, 3, 1, 1)
    else:
        mask = mask[:, :, None]

    imgs[:, 0:n_noisy] *= mask  # apply noise mask
    imgs[:, 0:n_noisy] += noise  # apply additive (Gaussian) noise
    imgs[:, 0:n_noisy] = imgs[:, 0:n_noisy].clamp_(min=0, max=1)
    return imgs


class EllipseNoiseTransform:
    def __init__(self, seed=None):
        self.seed = seed
        if seed:
            print("Init EllipseNoiseTransform with seed", seed)
            self.gen = torch.Generator()
            self.gen.manual_seed(seed)
        else:
            self.gen = None

    def reset_random_generator(self):
        self.gen.manual_seed(self.seed)

    def __call__(self, img):
        # img: torch tensor (3, 64, 64), float32

        n_noisy = cfg.args.n_noisy

        # imgs = torch.zeros((n_noisy + 1, img.size(0), img.size(1), img.size(2)))

        radius = torch.randint(low=cfg.noise.radius.low,
                               high=cfg.noise.radius.high,
                               size=(
                                   img.shape[0], cfg.noise.n_ellipses, n_noisy),
                               generator=self.gen)
        center = torch.randint(low=1, high=cfg.ds.res - 2,
                               size=(
                                   img.shape[0], cfg.noise.n_ellipses, n_noisy,
                                   2),
                               generator=self.gen)
        ellip = torch.rand(size=(img.shape[0], cfg.noise.n_ellipses, n_noisy),
                           generator=self.gen) + 0.5
        # translation = torch.randint(low=-cfg.noise.translation.abs, high=cfg.noise.translation.abs, size=(2, ), generator=self.gen)
        gaussian_noise = cfg.noise.gaussian_var * torch.randn(
            size=(img.shape[0], n_noisy, 1, img.shape[-2], img.shape[-1]),
            generator=self.gen) if cfg.noise.gaussian_var else torch.tensor(0)

        # imgs[-1] = img
        mask = create_elliptic_mask(size=img.shape[-1], center=center,
                                    radius=radius,
                                    ellip=ellip)  # (n_ellipses=4, n_noisy=3, size=64, size=64])

        return apply_mask_and_noise(mask, gaussian_noise, img, n_noisy,
                                    n_channels=cfg.ds.n_channels)  # (4, 1, 64, 64)


if __name__ == '__main__':
    transform = EllipseNoiseTransform()

    img = torch.ones(size=(1, 28, 28)) * 0.5  # create gray example image

    img_transformed = transform(img)

    # visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, cfg.args.n_noisy)
    for i in range(cfg.args.n_noisy):
        axes[i].imshow(img_transformed[i, 0], cmap='gray')

    plt.imshow(img_transformed[0, 0], cmap='gray')
    plt.show()
