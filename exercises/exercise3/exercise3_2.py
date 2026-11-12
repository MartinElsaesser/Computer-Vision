import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, nn, np, plt, torch, transforms


@app.function
def checked_cast[CastT](typ: type[CastT], val: object) -> CastT:
    """
    Runtime-check an object for a specific type and return it cast as such
    """
    if not isinstance(val, typ):
        msg = f"{val}: expected {typ.__name__}, not {val.__class__.__name__}"
        raise TypeError(msg)
    return val


@app.cell
def _(np):
    # Sobel Filter
    Gx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=np.float32
    )

    Gy = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=np.float32
    )
    return


@app.cell
def _():
    from PIL import Image, ImageOps
    import os
    import cv2
    import os

    img_path = os.path.abspath(
        os.path.join(os.getcwd(), "../../datasets/SchwarzerKreis.png")
    )
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img
    return (img,)


@app.cell
def _(mo):
    options = ["horizontal", "vertical"]
    direction_selector = mo.ui.multiselect(options=options, max_selections=2, value=["horizontal"])
    return (direction_selector,)


@app.cell
def _(direction_selector, mo):
    mo.hstack([direction_selector, mo.md(f"Selected directions: {direction_selector.value}")])
    return


@app.cell
def _(nn, np, torch):
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        padding_mode="replicate",
    )
    matrix = np.matrix([1, 2, 1]).T * np.matrix([1,2,1])
    binomial_kernel = torch.tensor(matrix, dtype=torch.float32).view(1,1,3,3)
    binomial_kernel_scale_factor = np.sum([1,2,1]) ** 2
    binomial_kernel, binomial_kernel_scale_factor
    return binomial_kernel, binomial_kernel_scale_factor, conv2d


@app.cell
def _(binomial_kernel, conv2d):
    conv2d.load_state_dict({"weight": binomial_kernel})
    conv2d.state_dict()
    return


@app.cell
def _(binomial_kernel_scale_factor, conv2d, img, torch, transforms):
    img_tensor =  transforms.ToTensor()(img)
    img_blured: torch.Tensor = conv2d(img_tensor)
    img_blured_scaled: torch.Tensor = img_blured / binomial_kernel_scale_factor  * 255
    img_blured_scaled
    return (img_blured_scaled,)


@app.cell
def _(img_blured_scaled: "torch.Tensor", plt):
    plt.imshow(img_blured_scaled.squeeze().detach().numpy(), cmap="gray")
    plt.show()
    return


@app.cell
def _(torch):

    GxWeights = torch.tensor([
        [-0.25, 0, 0.25],
        [-0.5, 0, 0.5],
        [-0.25, 0, 0.25],
    ])
    GyWeights = torch.tensor([
        [-0.25, -0.5, -0.25],
        [0,0,0],
        [0.25,0.5,0.25],
    ])

    weights = torch.stack([ GxWeights, GyWeights ], dim=0).unsqueeze(1)
    weights
    return (weights,)


@app.cell
def _(nn, weights):
    conv2d_sobel = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode="replicate",)


    state_dict = conv2d_sobel.state_dict()
    state_dict["weight"] = weights
    conv2d_sobel.load_state_dict(state_dict)
    return (conv2d_sobel,)


@app.cell
def _(img_blured_scaled: "torch.Tensor"):
    img_blured_scaled.shape
    return


@app.cell
def _(conv2d_sobel, img_blured_scaled: "torch.Tensor"):
    img_filtered = conv2d_sobel(img_blured_scaled)
    img_filtered_X = img_filtered[0]
    img_filtered_y = img_filtered[1]
    img_filtered_y
    return img_filtered_X, img_filtered_y


@app.cell
def _(img_filtered_X, img_filtered_y, plt):
    plt.imshow(img_filtered_X.squeeze().detach().numpy(), cmap="gray")
    plt.show()

    plt.imshow(img_filtered_y.squeeze().detach().numpy(), cmap="gray")
    plt.show()
    return


@app.cell
def _(img_filtered_X, img_filtered_y, plt, torch):
    magnitude = torch.sqrt(img_filtered_X**2 + img_filtered_y**2)
    magnitude.shape

    plt.imshow(magnitude.detach().numpy(), cmap="gray")
    return (magnitude,)


@app.cell
def _(img_filtered_X, img_filtered_y, torch):
    angle_in_radians = torch.atan2(img_filtered_y, img_filtered_X)
    return (angle_in_radians,)


@app.cell
def _(angle_in_radians, magnitude, np, plt):
    # Move to numpy (detach in case they come from an autograd graph)
    ang = angle_in_radians.detach().cpu().numpy()
    mag = magnitude.detach().cpu().numpy()

    # Normalize magnitude for arrow lengths (robust to outliers)
    mag_norm = mag / (np.percentile(mag, 99) + 1e-8)
    mag_norm = np.clip(mag_norm, 0, 1)

    # Subsample for readability
    step = max(1, min(mag.shape) // 40)
    ys, xs = np.mgrid[0:mag.shape[0]:step, 0:mag.shape[1]:step]
    u = np.cos(ang[::step, ::step]) * mag_norm[::step, ::step]
    v = np.sin(ang[::step, ::step]) * mag_norm[::step, ::step]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mag, cmap="gray", alpha=0.6)
    ax.quiver(xs, ys, u, v, color="red", pivot="mid",
              angles="xy", scale_units="xy", scale=0.5)
    ax.invert_yaxis()  # match image row-0-at-top convention
    ax.set_title("Gradient direction field (subsampled)")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
