import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO, YOLO_ONNX
import os
from tqdm import tqdm

def predict_images_in_folder(yolo, folder_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_names = os.listdir(folder_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, img_name)
            try:
                image = Image.open(image_path)
            except:
                print('Error opening image: {}'.format(img_name))
                continue
            else:
                r_image = yolo.detect_image(image)
                save_image_path = os.path.join(save_path, img_name.replace(".jpg", ".png"))
                r_image.save(save_image_path, quality=95, subsampling=0)

if __name__ == "__main__":
    mode = "dir_predict"
    crop = False
    count = False
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    test_interval = 100
    fps_image_path = "img/street.jpg"
    dir_origin_path = "F:/hjy/project/yolov7-pytorch-master/yolov7-pytorch-master/test"
    dir_save_path = "F:/hjy/project/yolov7-pytorch-master/yolov7-pytorch-master/testout"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video). Please check if the camera is installed correctly (or if the video path is correct).")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + ' FPS, @batch_size 1')

    elif mode == "dir_predict":
        predict_images_in_folder(yolo, dir_origin_path, dir_save_path)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
