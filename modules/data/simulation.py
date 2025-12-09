import numpy as np
from typing import Tuple, Union

# ============================================================================
# SIM RECONSTRUCTION CLASSES & FUNCTIONS
# ============================================================================

class OTF:
    """
    The Optical Transfer Function of the optical system.
    """
    def __init__(self, na: float, wavelength: float, pixel_size: float,
                 image_size: int, curvature: float):
        cutoff_frequency = 1000 * 2 * na / wavelength
        self.image_cutoff = cutoff_frequency * pixel_size * image_size
        self.image_size = image_size
        self.curvature = curvature

    def __call__(self, size: int = None, x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        if size is None:
            size = self.image_size
        x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                           np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
        distance_to_origin = np.hypot(x + x_shift, y + y_shift)
        return self.value(np.minimum(distance_to_origin / self.image_cutoff, 1))

    def value(self, x):
        return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x * x)) * self.curvature ** x

def illumination_pattern(angle, frequency, phase_offset, amplitude, size) -> np.ndarray:
    n = size // 2
    Y, X = np.mgrid[-n:n, -n:n]
    ky, kx = np.sin(angle) * frequency, np.cos(angle) * frequency
    return 1 + amplitude * np.cos(2 * np.pi * (X * kx + Y * ky) + phase_offset)

class PerlinNoise:
    def __init__(self, size: int, res: int):
        meshgrid = np.mgrid[0:res:res / size, 0:res:res / size]
        self.grid = np.stack(meshgrid) % 1
        self.t = self._fade(self.grid)
        self.d = size // res
        self.sample_size = (res + 1, res + 1)

    def __call__(self) -> np.array:
        angles = 2 * np.pi * np.random.random_sample(self.sample_size)
        gradients = np.dstack((np.cos(angles), np.sin(angles))).repeat(self.d, 0).repeat(self.d, 1)
        n00 = (np.dstack((self.grid[0], self.grid[1])) * gradients[:-self.d, :-self.d]).sum(2)
        n10 = (np.dstack((self.grid[0] - 1, self.grid[1])) * gradients[self.d:, :-self.d]).sum(2)
        n01 = (np.dstack((self.grid[0], self.grid[1] - 1)) * gradients[:-self.d, self.d:]).sum(2)
        n11 = (np.dstack((self.grid[0] - 1, self.grid[1] - 1)) * gradients[self.d:, self.d:]).sum(2)
        n0 = n00 * (1 - self.t[0]) + n10 * self.t[0]
        n1 = n01 * (1 - self.t[1]) + n11 * self.t[1]
        return 0.5 + 2 ** -0.5 * (n0 * (1 - self.t[1]) + n1 * self.t[1])

    @staticmethod
    def _fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

def _component_separation_matrix(phase_offset: float = 0, modulation: float = 1) -> np.ndarray:
    phases = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) + phase_offset
    M = np.ones((3, 3), dtype=np.complex128)
    M[:, 1] = 0.5 * modulation * (np.cos(phases) + 1j * np.sin(phases))
    M[:, 2] = 0.5 * modulation * (np.cos(-phases) + 1j * np.sin(-phases))
    return np.linalg.inv(M)

def separate_components(fft_images: np.ndarray, phase_offsets=np.zeros(3), phase_modulations=np.ones(3)) -> np.ndarray:
    separation_matrices = np.array([_component_separation_matrix(phase_offsets[i], phase_modulations[i]) for i in range(3)])
    return np.einsum('aij,ajkl->aikl', separation_matrices, fft_images.reshape(3, 3, *fft_images.shape[1:])).reshape(9, *fft_images.shape[1:])

def fourier_shift(fft_image: np.ndarray, x_shift: Union[float, np.ndarray], y_shift: Union[float, np.ndarray]) -> np.ndarray:
    height, width = fft_image.shape[-2:]
    spatial_image = np.fft.ifft2(fft_image)
    x_indices = np.arange(-width // 2, width // 2, dtype=int)
    y_indices = np.arange(-height // 2, height // 2, dtype=int)
    x = np.exp(-1j * 2 * np.pi * x_shift * x_indices[None, :] / width)
    y = np.exp(-1j * 2 * np.pi * y_shift * y_indices[:, None] / height)
    shifted_spatial_image = spatial_image * (x * y)
    return np.fft.fft2(shifted_spatial_image)

def shift_components(components: np.ndarray, shifts) -> None:
    for i in range(3):
        components[i * 3 + 1, ...] = fourier_shift(components[i * 3 + 1, ...], shifts[i][1], shifts[i][0])
        components[i * 3 + 2, ...] = fourier_shift(components[i * 3 + 2, ...], -shifts[i][1], -shifts[i][0])

def pad_components(components: np.ndarray) -> np.ndarray:
    size = components.shape[-1]
    padded_components = np.zeros(components.shape[:-2] + (size * 2, size * 2), dtype=np.complex128)
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]),
                       np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]))
    padded_components[..., y, x] = components
    return padded_components

def map_otf_support(components: np.ndarray, shifts, otf: OTF) -> None:
    size = components.shape[-1]
    components[::3] *= np.conjugate(otf(size))
    for i in range(3):
        components[i * 3 + 1] *= np.conjugate(otf(size, shifts[i][1], shifts[i][0]))
        components[i * 3 + 2] *= np.conjugate(otf(size, -shifts[i][1], -shifts[i][0]))

def wiener_filter(shifts, otf: OTF, w: float, size: int) -> np.ndarray:
    otf0 = np.abs(otf(size)) ** 2
    wiener = 3 * otf0
    for i in range(3):
        otf1 = np.abs(otf(size, shifts[i][1], shifts[i][0])) ** 2
        otf2 = np.abs(otf(size, -shifts[i][1], -shifts[i][0])) ** 2
        wiener += otf1 + otf2
    return 1 / (wiener + w * w)

def apodization_filter(dist_ratio: float, bend: float, size: int) -> np.ndarray:
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                       np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
    distance = np.hypot(x, y) * dist_ratio
    mask = np.bitwise_and(0 <= distance, distance < 1)
    apo = np.power((2 / np.pi) * (np.arccos(distance, where=mask) - distance * np.sqrt(1 - distance * distance, where=mask)), bend, where=mask)
    return np.where(mask, apo, 0)

def run_reconstruction(fft_images: np.ndarray, otf: OTF, shifts, phase_offsets, modulations, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    size = fft_images.shape[-1]
    cutoff = 1000 * 2 * config["na"] / config["wavelength"]
    apo_dist_ratio = 1 / (config["px_size"] * config["apo_cutoff"] * cutoff * size)
    components = separate_components(fft_images, phase_offsets, modulations)
    components = pad_components(components)
    shift_components(components, shifts)
    map_otf_support(components, shifts, otf)
    wiener = wiener_filter(shifts, otf, config["wiener_parameter"], size * 2)
    apodization = apodization_filter(apo_dist_ratio, config["apo_bend"], size * 2)
    fft_result = np.sum(components, 0) * wiener * apodization
    spatial_result = np.real(np.fft.ifft2(fft_result))
    return fft_result, spatial_result

class SyntheticDataset:
    """Base synthetic dataset for SIM images."""
    def __init__(self, contrast_fg_range=(0.0, 1.0), contrast_bg_range=(0.0, 1.0)):
        self.patch_size = 128
        self.contrast_fg_range = contrast_fg_range
        self.contrast_bg_range = contrast_bg_range
        self.frequency = 0.17
        self.amplitude = 1.0
        self.config = {
            "na": 1.49, "wavelength": 512, "px_size": 0.07,
            "wiener_parameter": 0.1, "apo_cutoff": 2.0, "apo_bend": 0.9
        }
        self.otf = OTF(self.config['na'], self.config['wavelength'], self.config['px_size'], self.patch_size // 2, 0.3)
        self.otf_mult = self.otf(self.patch_size)
        self.perlin = PerlinNoise(self.patch_size, 1)

    def _simulate_sim(self, image):
        angle0 = np.random.uniform(0, np.pi * 2)
        phase_offsets = np.random.uniform(0, np.pi * 2, 3)
        shifts = [(self.frequency * self.patch_size * np.sin(angle0 + i * np.pi / 3),
                   self.frequency * self.patch_size * np.cos(angle0 + i * np.pi / 3)) for i in range(3)]
        illumination = np.stack([illumination_pattern(angle0 + i // 3 * np.pi / 3, self.frequency,
                                                      phase_offsets[i // 3] + (i % 3) * np.pi * 2 / 3,
                                                      self.amplitude, self.patch_size) for i in range(9)])
        fg_c = np.random.uniform(*self.contrast_fg_range)
        bg_c = np.random.uniform(*self.contrast_bg_range)
        foreground = 250 + fg_c * 500
        background = 50 + bg_c * 50
        high_res_image = (image * foreground + background) * self.perlin()
        ix = np.fft.fft2(illumination * high_res_image)
        hix = self.otf_mult * ix
        dhix = hix.reshape(9, 2, self.patch_size // 2, 2, self.patch_size // 2).sum((1, 3)) / 4
        poisson_input = np.maximum(0, np.fft.ifft2(dhix).real)
        low_res_images = np.random.poisson(poisson_input).astype(np.float64)
        noisy_shifts = [np.random.triangular((y - 0.25, x - 0.25), (y, x), (y + 0.25, x + 0.25)) for y, x in shifts]
        noisy_phase_offsets = np.random.normal(phase_offsets, np.pi / 6)
        noisy_amplitudes = np.random.normal(self.amplitude, 0.1, 3)
        reconstruction = run_reconstruction(np.fft.fft2(low_res_images), self.otf, noisy_shifts,
                                          noisy_phase_offsets, noisy_amplitudes, self.config)[1]
        return (reconstruction - np.mean(reconstruction)) / np.std(reconstruction)
