import numpy as np
import sys
import dlib
import os
import cv2
import math
from PIL import Image
import multiprocessing
from joblib import Parallel, delayed
from os import path, makedirs


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


def process_image(f, save_path, base_path, detector, predictor, padding):
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
        # print ("image already exists, skipping.")
        return

    if os.path.isfile(f):

        img = cv2.imread(f)
        if img is not None:
            if img.shape[0] > 30 and img.shape[1] > 30:
                faces = []
                img_small = None
                if img.shape[0] > 512 and img.shape[1] > 512:
                    rfactor = 0.25
                    img_small = cv2.resize(img, (0, 0), fx=rfactor, fy=rfactor)
                    faces = detector(img_small, 1)

                if len(faces) == 0 or img_small is None:
                    faces = detector(img, 1)
                    rfactor = 1
                # print("Number of faces detected: {}".format(len(faces)))

                ind = -1
                max_ar = -1
                if len(faces) > 0:
                    for temp in range(0, len(faces)):
                        area = (int(faces[temp].bottom()) - int(faces[temp].top())) * \
                               (int(faces[temp].right()) - int(faces[temp].left()))
                        if area > max_ar:
                            max_ar = area
                            ind = temp

                    d = faces[ind]
                    d = dlib.rectangle(int(d.left() * (1 / rfactor)), int(d.top() * (
                        1 / rfactor)), int(d.right() * (1 / rfactor)), int(d.bottom() * (1 / rfactor)))
                    shape = predictor(img, d)

                    # get eye center of each image
                    img_left_center = [(float(str(shape.part(36)).split(",")[0][1:]) + float(str(shape.part(39)).split(",")[
                                        0][1:])) / 2, (float(str(shape.part(36)).split(",")[1][:-1]) + float(str(shape.part(39)).split(",")[1][:-1])) / 2]

                    img_right_center = [(float(str(shape.part(42)).split(",")[0][1:]) + float(str(shape.part(45)).split(",")[
                                         0][1:])) / 2, (float(str(shape.part(42)).split(",")[1][:-1]) + float(str(shape.part(45)).split(",")[1][:-1])) / 2]
                    img_cropped = CropFace(
                        img, img_left_center, img_right_center, (offset, offset), (sz, sz), padding)
                    strng = img_name.split('.')[0] + '.jpg'
                    cv2.imwrite(os.path.join(save_path, strng),
                                np.asarray(img_cropped))

                else:
                    print ("Cannot find a face in the image {}!".format(f))
                    error_path = save_path + '_error'

                    if not path.exists(error_path):
                        makedirs(error_path)

                    img_error_path = path.join(error_path, img_name)
                    cv2.imwrite(img_error_path, img)

        else:
            print('Cannot load image {}'.format(img_name))
    else:
        print ("warning: cannot find original file!")


def LandmarkImage(predictor, faces_folder_path, save_path):

    detector = dlib.get_frontal_face_detector()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_cores = 1  # multiprocessing.cpu_count()
    filepaths = []
    for root, dirs, files in os.walk(faces_folder_path):
        for file in files:
            if (file.endswith('.png') or file.endswith('.JPG') or file.endswith('.jpg')) and not file.startswith("."):
                filepaths.append(os.path.join(root, file))
    base_path = faces_folder_path

    Parallel(n_jobs=num_cores)(delayed(process_image)(f, save_path,
                                                      base_path, detector, predictor, .20) for f in filepaths)


faces_folder_path = sys.argv[1]
predictor_path = sys.argv[2]
save_path = sys.argv[3]
sz = int(sys.argv[4])
offset = float(sys.argv[5])
predictor = dlib.shape_predictor(predictor_path)
LandmarkImage(predictor, faces_folder_path, save_path)
# pool = multiprocessing.Pool(processes=processors)
