import numpy as np
import argparse
from os import path, makedirs
import dlib
import cv2
from PIL import Image
from scipy.spatial import distance
import sys
sys.path.insert(0, '../../insightface/deploy/')
import face_model


def crop_faces(model1, model2, cnn_face_detector, img_list_path, source, destination, image_size):
    img_list = np.loadtxt(img_list_path, dtype=np.str)

    for image_name in img_list:
        image_path = path.join(source, image_name)
        output_path = path.join(destination, image_name)

        if path.isfile(output_path):
            continue

        img_orig = cv2.imread(image_path)

        if img_orig is None:
            print ("Cannot load image {}!".format(image_name))
            continue

        img = np.copy(img_orig)

        if img.shape[0] > 2048 or img.shape[1] > 2048:
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        if img.shape[0] < image_size or img.shape[0] < image_size:
            proportion = max(img.shape[0], img.shape[1]) / min(img.shape[0], img.shape[1])

            if img.shape[0] < img.shape[1]:
                height = image_size
                width = round(image_size * proportion)
            else:
                width = image_size
                height = round(image_size * proportion)

            img = cv2.resize(img, (width, height))

        # detect from beggining
        face_aligned = model1.get_input(img)

        if face_aligned is not None:
            face_aligned = np.transpose(face_aligned, (1, 2, 0))
            face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

            if not path.exists(path.split(output_path)[0]):
                makedirs(path.split(output_path)[0])

            cv2.imwrite(output_path, face_aligned)

        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = cnn_face_detector(img_rgb, 1)

            if len(faces) > 0:
                ind = -1

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

                # other datasets
                max_ar = -1

                for temp in range(0, len(faces)):
                    area = (int(faces[temp].rect.bottom()) - int(faces[temp].rect.top())) * \
                           (int(faces[temp].rect.right()) - int(faces[temp].rect.left()))

                    if area > max_ar:
                        max_ar = area
                        ind = temp

                d = faces[ind]

                left = int(d.rect.left() * 1)
                top = int(d.rect.top() * 1)
                right = int(d.rect.right() * 1)
                bottom = int(d.rect.bottom() * 1)

                padding_px_x = int((right - left) * 0.5)
                padding_px_y = int((bottom - top) * 0.5)

                image = Image.fromarray(img)

                face = image.crop((max(0, int(left) - padding_px_x),
                                   max(0, int(top) - padding_px_y),
                                   min(img.shape[1], int(right) + padding_px_x),
                                   min(img.shape[0], int(bottom) + padding_px_y)))

                face_arr = np.asarray(face)

                # detect using bbox
                face_aligned = model2.get_input(face_arr)

                if face_aligned is not None:
                    face_aligned = np.transpose(face_aligned, (1, 2, 0))
                    face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)

                    if not path.exists(path.split(output_path)[0]):
                        makedirs(path.split(output_path)[0])

                    cv2.imwrite(output_path, face_aligned)

                else:
                    error_path = path.join(destination[:-1] + '_error', image_name)

                    if not path.exists(path.split(error_path)[0]):
                        makedirs(path.split(error_path)[0])

                    cv2.imwrite(error_path, img_orig)

                    error_path = path.join(destination[:-1] + '_error_size', image_name)

                    if not path.exists(path.split(error_path)[0]):
                        makedirs(path.split(error_path)[0])

                    img = cv2.resize(img_orig, (image_size, image_size))

                    cv2.imwrite(error_path, img)
                    print ("Cannot find a face in the image {}!".format(image_name))

            else:
                error_path = path.join(destination[:-1] + '_error', image_name)

                if not path.exists(path.split(error_path)[0]):
                    makedirs(path.split(error_path)[0])

                cv2.imwrite(error_path, img_orig)

                error_path = path.join(destination[:-1] + '_error_size', image_name)

                if not path.exists(path.split(error_path)[0]):
                    makedirs(path.split(error_path)[0])

                img = cv2.resize(img_orig, (image_size, image_size))

                cv2.imwrite(error_path, img)
                print ("Cannot find a face in the image {}!".format(image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and crop faces with MTCNN')
    parser.add_argument('--dlib_model', '-m', help='Dlib face detector model.',
                        default='./dlib_models/mmod_human_face_detector.dat')
    parser.add_argument('--img_list', '-i', help='File with an image list.')
    parser.add_argument('--source', '-s', help='Path for the images.')
    parser.add_argument('--dest', '-d', help='Folder to save the cropped faces.')
    parser.add_argument('--image_size', '-is', help='Size to resize.', default='112,112')

    # InsightFace Face Model
    parser.add_argument('--model', help='path to model.', default='')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gender_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn: 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    args = parser.parse_args()

    # uses higher thresholds to detect at first with
    model1 = face_model.FaceModel(args)

    # if fails use Dlib CNN to crop the face and MTCNN without thresholds to align
    args.det = 1
    model2 = face_model.FaceModel(args)

    if not path.exists(args.dest):
        makedirs(args.dest)

    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.dlib_model)

    crop_faces(model1, model2, cnn_face_detector, args.img_list, args.source, args.dest,
               int(args.image_size.split(',')[0]))
