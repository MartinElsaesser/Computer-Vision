import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import cv2
    import os
    import glob
    import matplotlib.pyplot as plt

    img = cv2.imread('/workspace/datasets/SIFT_images/ImageCalendar.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img
    return cv2, img, mo, plt


@app.cell
def _(mo):
    kernel_size_slider = mo.ui.slider(start=1, stop=20, label="Kernel Size", value=3, show_value=True)
    options = ["mean", "binomial"]
    method_selector = mo.ui.multiselect(options=options, max_selections=1, value=["mean"])
    return kernel_size_slider, method_selector


@app.cell
def _(kernel_size_slider, method_selector, mo):
    mo.vstack([
        mo.hstack([kernel_size_slider, mo.md(f"Kernel Size: {kernel_size_slider.value}x{kernel_size_slider.value}")]), 
        mo.hstack([method_selector, mo.md(f"Selected Value: {method_selector.value}")])
    ])
    return


@app.cell
def _(kernel_size_slider, method_selector):
    from typing import cast
    selected_method = method_selector.value[0]
    kernel_size = cast(int, kernel_size_slider.value)
    selected_method, kernel_size
    return kernel_size, selected_method


@app.cell
def _(kernel_size, selected_method):
    import math
    import numpy as np
    if selected_method == "mean":
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
    elif selected_method == "binomial": 
        kernel = np.matrix([ math.comb(kernel_size - 1, n) for n in range(kernel_size)])
        kernel = kernel.T * kernel
        kernel = kernel / np.sum(kernel[0])**2
        kernel = np.array(kernel)
    kernel
    return kernel, np


@app.cell
def _(cv2, img, kernel, np, plt):
    res = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    res = res.astype(np.uint8)

    figure = plt.figure(figsize=(10,8))
    plt.imshow(res, cmap="gray")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
