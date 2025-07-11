import cv2
import numpy as np
import torch


def extract_noise(images, filter_size=3):

    if isinstance(images, torch.Tensor):
        b, c, h, w = images.shape
    elif isinstance(images, np.ndarray):
        b, c, h, w = images.shape
        images = torch.tensor(images, dtype=torch.float32)  # 转换为torch.Tensor
    else:
        raise ValueError("Input should be either a numpy.ndarray or torch.Tensor")

    assert c == 3, "Input images must have 3 channels (RGB)."

    noise = torch.zeros_like(images, dtype=torch.float32)

    if images.is_cuda:
        images = images.cpu()

    for i in range(b):
        image_rgb = images[i].numpy().transpose(1, 2, 0)  # transform h,w,c

        image_filtered = np.zeros_like(image_rgb)
        for j in range(3):  # RGB 3 channels
            image_filtered[..., j] = cv2.medianBlur(image_rgb[..., j], filter_size)

        # noise
        noise_np = np.abs(image_rgb - image_filtered)

        # save
        noise[i] = torch.tensor(noise_np.transpose(2, 0, 1), dtype=torch.float32)

    # [0, 1]
    for channel in range(c):
        channel_noise = noise[:, channel, :, :]
        noise_min = channel_noise.min()
        noise_max = channel_noise.max()
        if noise_max > noise_min:
            noise[:, channel, :, :] = (channel_noise - noise_min) / (noise_max - noise_min)
        else:
            noise[:, channel, :, :] = torch.zeros_like(channel_noise)

    return noise

def process_image(image):
    """
    Process an image of shape (b, 3, h, w) to compute pairwise channel differences
    and normalize the result per channel.

    Args:
        image: PyTorch tensor of shape (b, 3, h, w)

    Returns:
        output: PyTorch tensor of shape (b, 3, h, w) with normalized channel differences
    """
    # Check input shape
    if len(image.shape) != 4 or image.shape[1] != 3:
        raise ValueError("Input image must have shape (b, 3, h, w)")

    # Get dimensions
    b, _, h, w = image.shape

    # Initialize output tensor for differences: R-G, G-B, B-R
    differences = torch.zeros((b, 3, h, w), dtype=torch.float32, device=image.device)

    # Compute pairwise channel differences
    differences[:, 0, :, :] = image[:, 0, :, :] - image[:, 1, :, :]  # R-G
    differences[:, 1, :, :] = image[:, 1, :, :] - image[:, 2, :, :]  # G-B
    differences[:, 2, :, :] = image[:, 2, :, :] - image[:, 0, :, :]  # B-R

    # Normalize each channel to [0, 1]
    output = torch.zeros_like(differences)
    for i in range(3):
        channel = differences[:, i, :, :]
        # Compute min and max over height and width dimensions
        channel_min = torch.amin(channel, dim=(1, 2), keepdim=True)
        channel_max = torch.amax(channel, dim=(1, 2), keepdim=True)
        channel_range = channel_max - channel_min
        # Handle case where range is zero
        channel_range = torch.where(channel_range == 0, torch.tensor(1.0, device=image.device), channel_range)
        output[:, i, :, :] = (channel - channel_min) / channel_range

    return output


# Example usage
if __name__ == "__main__":
    # Create a sample image (batch_size=2, channels=3, height=64, width=64)
    sample_image = np.random.rand(2, 3, 64, 64)

    # Process the image
    result = process_image(sample_image)

    # Verify output shape and value range
    print("Output shape:", result.shape)
    print("Output min:", result.min())
    print("Output max:", result.max())