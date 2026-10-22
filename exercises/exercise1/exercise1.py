import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # importing os module
    import os

    path2ImgFiles = os.path.join(os.getcwd(), "datasets/pedestrian/PNGImages")
    print('path:', path2ImgFiles)

    dirExists = os.path.exists(path2ImgFiles)
    if not dirExists:
        print('Laden Sie sich das Pedestrian Dataset von der Moodle Seite herunter')
    else:
        print('Verzeichnis vorhanden')

    import cv2
    import matplotlib.pyplot as plt


    list_of_image_files = os.listdir(path2ImgFiles)
    list_of_image_files
    return list_of_image_files, path2ImgFiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Erstes Bild einladen
    """)
    return


@app.cell
def _(list_of_image_files, path2ImgFiles):
    fname = path2ImgFiles + "/" + list_of_image_files[0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
