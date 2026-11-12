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
    return Gx, Gy


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
    return cv2, img


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
def _(Gx, Gy, cv2, direction_selector, img, np):
    scale_factor = 1
    img2Show: np.typing.NDArray = np.array([])
    if "horizontal" in direction_selector.value:
        G_x = cv2.filter2D(img, cv2.CV_32F, Gx)
        img2Show = G_x
        scale_factor = np.sum(np.abs(Gx))

    if "vertical" in direction_selector.value:
        G_y = cv2.filter2D(img, cv2.CV_32F, Gy)
        img2Show = G_y
        scale_factor = np.sum(np.abs(Gy))

    if "horizontal" in direction_selector.value and "vertical" in direction_selector.value:
        G_x = cv2.filter2D(img, cv2.CV_32F, Gx)
        G_y = cv2.filter2D(img, cv2.CV_32F, Gy)
        img2Show = np.sqrt(G_x**2 + G_y**2)
        scale_factor = np.sum(np.abs(Gy)) # TODO: scale_factor

    print(scale_factor)
    return img2Show, scale_factor


@app.cell
def _():
    offset = 0
    return (offset,)


@app.cell
def _(img2Show: "np.typing.NDArray", offset, scale_factor):
    img2ShowScaled =  ( img2Show / scale_factor ) + offset
    img2ShowScaled
    return


@app.cell
def _(img, img2Show: "np.typing.NDArray", plt):
    figure = plt.figure(figsize=(10,8))
    cols, rows = 2, 1
    figure.add_subplot(rows, cols, 1)
    plt.title("Originalbild")
    plt.axis("off")
    plt.imshow(img, cmap="gray")

    figure.add_subplot(rows, cols, 2)
    plt.title("Kantenbild")
    plt.axis("off")
    plt.imshow(img2Show, cmap="gray")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
