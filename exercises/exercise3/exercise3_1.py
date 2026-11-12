import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, np, plt


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
    import os
    import cv2

    img_path = os.path.abspath(
        os.path.join(os.getcwd(), "../../datasets/SchwarzerKreis.png")
    )

    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
