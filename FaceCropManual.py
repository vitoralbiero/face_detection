import numpy as np
import sys
import dlib
import os
import cv2
import math
from PIL import Image
import multiprocessing
from joblib import Parallel, delayed

eye_1 = None
eye_2 = None
image = None


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None,
                         scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    pilImg = Image.fromarray(image)
    pilImg2 = pilImg.transform(
        pilImg.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

    return pilImg2


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0),
             offset_pct=(0.2, 0.2), dest_sz=(70, 70), padding=.15):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    padding_px_x = int(crop_size[0] * padding)
    padding_px_y = int(crop_size[1] * padding)
    image = image.crop((int(crop_xy[0]) - padding_px_x,
                        int(crop_xy[1]) - padding_px_y,
                        int(crop_xy[0] + crop_size[0]) + padding_px_x,
                        int(crop_xy[1] + crop_size[1]) + padding_px_y))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def click(event, x, y, flags, param):
    global eye_1, eye_2, image
    # grab references to the global variables
    if event == cv2.EVENT_LBUTTONDOWN:
        if eye_1 is None:
            eye_1 = [x, y]

        elif eye_2 is None:
            eye_2 = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


def reset_eyes():
    global eye_1, eye_2

    eye_1 = None
    eye_2 = None


def process_image(f, save_path, base_path, padding):
    global eye_1, eye_2, image
    # print("Processing file: {}".format(f))
    img_name = os.path.basename(f)
    img_dir = os.path.dirname(f)
    saveLocation = img_dir[len(base_path):]
    if len(saveLocation) > 0 and (saveLocation[0] == '/' or \
        saveLocation[0] == '\\'):
        saveLocation = saveLocation[1:]
    save_path = os.path.join(save_path, saveLocation)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.isfile(os.path.join(save_path, img_name[:-4] + '.jpg')):
        print ("image already exists, skipping.")
        return

    if os.path.isfile(f):
        img = cv2.imread(f)
        if img is not None:
            image = img.copy()

            scale = 1

            if image.shape[0] > 1024:
                scale = 0.25
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", image)
                key = cv2.waitKey(1) & 0xFF

                # if the 'r' key is pressed, reset the cropping region
                if key == ord("r"):
                    reset_eyes()
                    image = img.copy()

                # if the 'c' key is pressed, break from the loop
                elif key == ord("c") and eye_1 is not None and eye_2 is not None:
                    break

            eye_1[0] /= scale
            eye_1[1] /= scale
            eye_2[0] /= scale
            eye_2[1] /= scale

            img_cropped = CropFace(
                img, eye_1, eye_2, (offset, offset), (sz, sz), padding)
            strng = img_name.split('.')[0] + '.JPG'
            cv2.imwrite(os.path.join(save_path, strng),
                        np.asarray(img_cropped))

            img_cropped = CropFace(
                img, eye_1, eye_2, (0.2, 0.2), (sz, sz), padding)
            strng = img_name.split('.')[0] + '_lbp.JPG'
            cv2.imwrite(os.path.join(save_path, strng),
                        np.asarray(img_cropped))

            reset_eyes()

        else:
            print ("Cannot find a face in this image!")
    else:
        print ("warning: cannot find original file!")


def LandmarkImage(faces_folder_path, save_path):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_cores = 1  # multiprocessing.cpu_count()
    filepaths = []
    for root, dirs, files in os.walk(faces_folder_path):
        for file in files:
            if (file.endswith('.png') or file.endswith('.JPG')) and not file.startswith("."):
                filepaths.append(os.path.join(root, file))
    base_path = faces_folder_path

    Parallel(n_jobs=num_cores)(delayed(process_image)(f, save_path, base_path, .20) for f in filepaths)


faces_folder_path = sys.argv[1]
save_path = sys.argv[2]
sz = int(sys.argv[3])
offset = float(sys.argv[4])
LandmarkImage(faces_folder_path, save_path)
# pool = multiprocessing.Pool(processes=processors)
