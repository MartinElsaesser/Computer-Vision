import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # 0.1 OpenCV, Numpy und Matplot importierten

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    return cv2, np, plt


@app.cell
def _(cv2):
    # 0.2 Bild einladen
    img = cv2.imread("/workspace/datasets/SIFT_images/ImageCalendar.jpg")
    if img is None:
        raise TypeError("img is None")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img,)


@app.cell
def _(np):
    # 0.3 Kernel für Binomialfilter erstellen
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    kernel /= sum(kernel[0, :]) ** 2
    kernel
    return (kernel,)


@app.cell
def _(cv2, img, kernel):
    # 0.4 Bild filtern
    res = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    res
    return (res,)


@app.cell
def _(plt, res):
    # 0.5 Ergebnis anzeigen
    _figure = plt.figure(figsize=(10, 8))
    plt.imshow(res, cmap="gray")
    plt.title("3 x 3 Binomialfilter")
    plt.show()
    return


@app.cell
def _(cv2, np, res):
    # 0.6 Prewitt-Operator auf das weichgezeichnete Bild anwenden

    kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
    kernel1 /= 3
    res_Prewitt = cv2.filter2D(res, ddepth=-1, kernel=kernel1)
    return kernel1, res_Prewitt


@app.cell
def _(plt, res_Prewitt):
    # 0.7 Ergebnis anzeigen

    _figure = plt.figure(figsize=(10, 8))
    plt.imshow(res_Prewitt, cmap="gray")
    plt.title("3 x 3 Prewitt-Operator")
    plt.show()
    return


@app.cell
def _(cv2, kernel1, plt, res):
    # 0.8 Warum sehen wir nur die Kanten von links hell nach rechts dunkel?

    # 0.9 Wie sieht der Wertebereich der Ergebnismatrix res_Prewitt aus? Hier ein Hinweis:

    kernel2 = kernel1 * -1
    res_Prewitt2 = cv2.filter2D(res, ddepth=-1, kernel=kernel2)
    _figure = plt.figure(figsize=(10, 8))
    plt.imshow(res_Prewitt2, cmap="gray")
    plt.title("3 x 3 Prewitt-Operator Kanten dunkel -> hell")
    plt.show()
    return


@app.cell
def _(cv2, np, res):
    # 0.10 Zur Visualisierung skalieren und verschieben wir den Wertebereich der Ergebnismatrix
    _kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
    # Skalierung des Wertebereichs (von [-3*255, +3*255] nach [-127.5, +127.5] bzw [-128, +128]) erfolgt hier über die Kernelmatrix
    _kernel /= 6
    res_Prewitt3 = cv2.filter2D(res, ddepth=cv2.CV_32F, kernel=_kernel)
    min, max = res_Prewitt3.min(), res_Prewitt3.max()
    print(f"{min=} {max=}")
    res_Prewitt3 += 128
    return (res_Prewitt3,)


@app.cell
def _(plt, res_Prewitt3):
    _figure = plt.figure(figsize=(10, 8))
    plt.imshow(res_Prewitt3, cmap="gray")
    plt.title("3 x 3 Prewitt-Operator (Ergebnis skaliert und verschoben)")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
