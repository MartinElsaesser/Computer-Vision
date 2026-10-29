import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    model = YOLO("yolo26n.pt")
    images = ["https://ultralytics.com/images/bus.jpg"]
    yolo_predictions = model.predict(images)

    yolo_predictions
    return model, plt, yolo_predictions


@app.cell
def _(plt, yolo_predictions):
    for _result in yolo_predictions:
        plt.figure(figsize=(12,8))
        plt.imshow(_result.plot()[:,:,::-1])
        plt.show()

    return


@app.cell
def _():
    import os
    import glob
    img_path = os.path.abspath(
        os.path.join(os.getcwd(), "../../datasets/")
    )
    list_of_image_files = glob.glob('/workspace/datasets/pedestrian/PNGImages/*.png', root_dir="/")
    list_of_image_files
    return (list_of_image_files,)


@app.cell
def _(list_of_image_files, model, plt):
    import cv2
    plt.figure(figsize=(18, 7))

    rows = 5
    cols = 2
    n = rows * cols
    for fname_i, fname in enumerate(list_of_image_files[:n], start=1):
        img_bgr = cv2.imread(fname)

        _results = model.predict([img_bgr])
        for _result in _results:
        
            xyxy = _result.boxes.xyxy
            names = [_result.names[cls.item()] for cls in _result.boxes.cls.int()]  
            #print(f'{xyxy=}')
            # print(names)

            out = img_bgr.copy()
            for j,(xt1, yt1, xt2, yt2) in enumerate(xyxy):
                if names[j] == 'person':
                    x1 = int(xt1)
                    y1 = int(yt1)
                    x2 = int(xt2)
                    y2 = int(yt2)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            plt.subplot(rows, cols, fname_i)
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.title(f"{fname_i}\nDetektionen: {len(xyxy)}")
            plt.axis('off')

    plt.suptitle("YOLO-Personendetektion – erste 10 Bilder", y=0.98)
    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
