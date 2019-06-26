import numpy as np
import argparse
from os import path, makedirs
import dlib
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.spatial import distance


def crop_faces(cnn_face_detector, img_list_path, source, destination, padding, resize, image_size, square):
    img_list = np.loadtxt(img_list_path, dtype=np.str)

    for image_name in img_list:
        image_path = path.join(source, image_name)
        output_path = path.join(destination, image_name)

        if path.isfile(output_path):
            # print ("image already exists, skipping...")
            continue

        img_orig = cv2.imread(image_path)

        if img_orig is None:
            print ("Cannot load image {}!".format(image_name))
            continue

        img = np.copy(img_orig)

        if img.shape[0] > 2048 or img.shape[1] > 2048:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        if img.shape[0] < image_size or img.shape[0] < image_size:
            # print('Skipping image {} as it is too small'.format(image_name))
            # continue

            proportion = max(img.shape[0], img.shape[1]) / min(img.shape[0], img.shape[1])

            if img.shape[0] < img.shape[1]:
                height = image_size
                width = round(image_size * proportion)
            else:
                width = image_size
                height = round(image_size * proportion)

            img = cv2.resize(img, (width, height))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = cnn_face_detector(img_rgb, 1)

        if len(faces) > 0:
            ind = -1
            max_ar = -1

            # other datasets
            for temp in range(0, len(faces)):
                area = (int(faces[temp].rect.bottom()) - int(faces[temp].rect.top())) * \
                       (int(faces[temp].rect.right()) - int(faces[temp].rect.left()))

                if area > max_ar:
                    max_ar = area
                    ind = temp

            # VGGFace2 faces are centered, so if more than 1 face detected, get the most centered one
            # min_dist = float('inf')
            # img_x = img.shape[1] // 2
            # img_y = img.shape[0] // 2

            # for temp in range(0, len(faces)):
            #     face_y = (int(faces[temp].rect.bottom()) + int(faces[temp].rect.top())) // 2
            #     face_x = (int(faces[temp].rect.right()) + int(faces[temp].rect.left())) // 2

            #     dist = distance.euclidean([face_x, face_y], [img_x, img_y])

            #     if dist < min_dist:
            #         min_dist = dist
            #         ind = temp

            d = faces[ind]

            left = int(d.rect.left() * 1)
            top = int(d.rect.top() * 1)
            right = int(d.rect.right() * 1)
            bottom = int(d.rect.bottom() * 1)

            padding_px_x = int((right - left) * padding)
            padding_px_y = int((bottom - top) * padding)

            image = Image.fromarray(img)

            face = image.crop((max(0, int(left) - padding_px_x),
                               max(0, int(top) - padding_px_y),
                               min(img.shape[1], int(right) + padding_px_x),
                               min(img.shape[0], int(bottom) + padding_px_y)))

            face_arr = np.asarray(face)

            if resize:
                if square:
                    height = image_size
                    width = image_size
                else:
                    proportion = max(face_arr.shape[0], face_arr.shape[1]) / min(face_arr.shape[0], face_arr.shape[1])

                    if face_arr.shape[0] < face_arr.shape[1]:
                        height = image_size
                        width = round(image_size * proportion)
                    else:
                        width = image_size
                        height = round(image_size * proportion)

                face = face.resize((width, height), Image.ANTIALIAS)

            if not path.exists(path.split(output_path)[0]):
                makedirs(path.split(output_path)[0])

            cv2.imwrite(output_path, np.asarray(face))

        else:
            error_path = path.join(destination[:-1] + '_error', image_name)

            if not path.exists(path.split(error_path)[0]):
                makedirs(path.split(error_path)[0])

            cv2.imwrite(error_path, img_orig)

            error_path = path.join(destination[:-1] + '_error_size', image_name)

            if not path.exists(path.split(error_path)[0]):
                makedirs(path.split(error_path)[0])

            if resize:
                if square:
                    height = image_size
                    width = image_size
                else:
                    proportion = max(img_orig.shape[0], img_orig.shape[1]) / min(img_orig.shape[0], img_orig.shape[1])

                    if img_orig.shape[0] < img_orig.shape[1]:
                        height = image_size
                        width = round(image_size * proportion)
                    else:
                        width = image_size
                        height = round(image_size * proportion)

                img = cv2.resize(img_orig, (width, height))

            cv2.imwrite(error_path, img)
            print ("Cannot find a face in the image {}!".format(image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and crop faces with Dlib CNN')
    parser.add_argument('--model', '-m', help='Dlib face detector model.')
    parser.add_argument('--img_list', '-i', help='File with an image list.')
    parser.add_argument('--source', '-s', help='Path for the images.')
    parser.add_argument('--dest', '-d', help='Folder to save the cropped faces.')
    parser.add_argument('--padding', '-p', help='Padding around the face to add.', default=0.3)
    parser.add_argument('--resize', help='Resize the detected faces.', default=False, action='store_true')
    parser.add_argument('--square', help='Make both height and width equal.', default=False, action='store_true')
    parser.add_argument('--image_size', '-is', help='Size to resize.', default=256)

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.model)

    crop_faces(cnn_face_detector, args.img_list, args.source, args.dest,
               float(args.padding), args.resize, int(args.image_size), args.square)
