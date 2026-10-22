import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # importing os module
    import os

    path2ImgFiles = "/workspace/datasets/pedestrian/PNGImages"
    print("path:", path2ImgFiles)

    dirExists = os.path.exists(path2ImgFiles)
    if not dirExists:
        print("Laden Sie sich das Pedestrian Dataset von der Moodle Seite herunter")
    else:
        print("Verzeichnis vorhanden")

    list_of_image_files = os.listdir(path2ImgFiles)
    list_of_image_files
    return list_of_image_files, os, path2ImgFiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Erstes Bild einladen
    """)
    return


@app.cell
def _(list_of_image_files, path2ImgFiles):
    import cv2
    import matplotlib.pyplot as plt

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    img_name = path2ImgFiles + "/" + list_of_image_files[0]
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2, hog, img, plt


@app.cell
def _(img, plt):
    plt.figure(figsize=(18, 10))
    plt.imshow(img)
    plt.title("Bildanzeige")
    plt.show()
    return


@app.cell
def _(cv2, img):
    img2Show = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img2Show, imgGray


@app.cell
def _(hog, imgGray):
    (rects, weights) = hog.detectMultiScale(
        imgGray, winStride=(4, 4), padding=(8, 8), scale=1.2
    )

    print(f"{rects=}")
    print(f"{weights=}")
    return (rects,)


@app.cell
def _(cv2, img2Show, plt, rects):
    for _x, _y, _w, _h in rects:
        cv2.rectangle(img2Show, (_x, _y), (_x + _w, _y + _h), (0, 0, 255), 2)
    plt.imshow(img2Show)
    plt.axis("off")
    plt.show()

    return


@app.cell
def _(cv2, hog, list_of_image_files, os, path2ImgFiles, plt):
    plt.figure(figsize=(18, 7))
    n = 10
    rows = 2
    cols = 5
    for i, fname in enumerate(list_of_image_files[1 : n + 1], start=1):
        img_bgr = cv2.imread(os.path.join(path2ImgFiles, fname))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _rects, _weights = hog.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.2 
        )
        out = img_bgr.copy()
        for _x, _y, _w, _h in _rects:
            cv2.rectangle(out, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)

        plt.subplot(rows, cols, i)
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.title(f"{fname}\nDetektionen:{len(_rects)}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
