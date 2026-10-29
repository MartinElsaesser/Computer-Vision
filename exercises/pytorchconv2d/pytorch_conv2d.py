import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np

    return nn, np, torch


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
    GxWeights = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )
    return (GxWeights,)


@app.cell
def _(nn):
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        padding_mode="replicate",
    )
    return (conv2d,)


@app.cell
def _(conv2d):
    ## Im State Dictionary finden wir die Gewichte für die Faltung
    stateDict = conv2d.state_dict()
    print(f"{stateDict=}")
    return


@app.cell
def _(GxWeights, conv2d, torch):
    ## Diese Gewichte müssen wir mit de Filterkern Sobel ersetzen
    conv2d.load_state_dict({"weight": torch.tensor(GxWeights).reshape(1, 1, 3, 3)})
    conv2d.state_dict()
    return


@app.cell
def _():
    import os

    img_path = os.path.abspath(
        os.path.join(os.getcwd(), "../../datasets/SchwarzerKreis.png")
    )
    return (img_path,)


@app.cell
def _(img_path):
    ## Bild einladen
    from PIL import Image, ImageOps

    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img
    return (img,)


@app.cell
def _(img, torch):
    ## Bild in Tensor umwandeln
    from typing import cast
    import torchvision.transforms as transforms

    toTensor = transforms.ToTensor()
    imgTensor = checked_cast(torch.types.Tensor, toTensor(img))
    imgTensor.size()
    return cast, imgTensor


@app.cell
def _(conv2d, imgTensor, torch):
    ## Faltung ausführen
    imagSobelGx = checked_cast(torch.types.Tensor, conv2d(imgTensor))
    imagSobelGx.shape
    return (imagSobelGx,)


@app.cell
def _(cast, imagSobelGx, np):
    ## Ergebnisbild anzeigen
    import matplotlib.pyplot as plt

    img2Show = cast(np.typing.NDArray, imagSobelGx.detach().numpy()[0])
    return img2Show, plt


@app.cell
def _(img2Show, plt):
    plt.imshow(img2Show, cmap="gray")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
