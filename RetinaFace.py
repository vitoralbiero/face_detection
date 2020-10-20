import numpy as np
import argparse
from os import path, makedirs
import cv2
import sys
from tqdm import tqdm
sys.path.insert(0, '../../insightface/common/')
sys.path.insert(0, '../../insightface/RetinaFace/')
from retinaface import RetinaFace
import face_align


def adjust_bbox(bbox, im_shape, offset):
    if int(bbox[3]) - int(bbox[1]) > int(bbox[2]) - int(bbox[0]):
        diff = ((int(bbox[3]) - int(bbox[1])) - (int(bbox[2]) - int(bbox[0]))) // 2

        top = max(int(bbox[1]), 0)
        bottom = min(int(bbox[3]), im_shape[0])
        left = max(int(bbox[0]) - diff, 0)
        right = min(int(bbox[2]) + diff, im_shape[1])
    else:
        diff = ((int(bbox[2]) - int(bbox[0])) - (int(bbox[3]) - int(bbox[1]))) // 2

        top = max(int(bbox[1]) - diff, 0)
        bottom = min(int(bbox[3]) + diff, im_shape[0])
        left = max(int(bbox[0]), 0)
        right = min(int(bbox[2]), im_shape[1])

    top = max(top - (int((bottom - top) * offset)), 0)
    bottom = min(bottom + (int((bottom - top) * offset)), im_shape[0])
    left = max(left - (int((right - left) * offset)), 0)
    right = min(right + (int((right - left) * offset)), im_shape[1])

    return top, bottom, left, right


def get_norm_crop(detector, image_name, im, target_size, max_size, image_size, align_mode, offset):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # if im_size_min <= 224:
    #     return None, None

    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    bbox, landmark = detector.detect(im, threshold=0.8, scales=[im_scale])

    if bbox.shape[0] == 0:
        # return None, None
        bbox, landmark = detector.detect(im, threshold=0.5, scales=[
                                         im_scale * 0.75, im_scale, im_scale * 2.0])
        print('refine ', image_name, im.shape, bbox.shape, landmark.shape)

    nrof_faces = bbox.shape[0]
    if nrof_faces > 0:
        det = bbox[:, 0:4]
        img_size = np.asarray(im.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # bindex = np.argmax(bounding_box_size - offset_dist_squared *
            #                    2.0)  # some extra weight on the centering
            bindex = np.argmax(bounding_box_size)

        cropped = None
        _bbox = det[bindex, 0:4]
        top, bottom, left, right = adjust_bbox(_bbox, im_shape, offset)

        cropped = im[top:bottom, left:right]
        cropped = cv2.resize(cropped, (image_size, image_size))

        warped = []
        # for i in range(nrof_faces):
        _landmark = landmark[bindex]
        warped.append(face_align.norm_crop(im, landmark=_landmark,
                                           image_size=image_size, mode=align_mode))

        return warped, cropped
    else:
        return None, None


def crop_faces(detector, img_list_path, source, destination, target_size,
               max_size, image_size, align_mode, offset):
    img_list = np.loadtxt(img_list_path, dtype=np.str)

    for image_name in tqdm(img_list):
        image_path = path.join(source, image_name)
        output_path = path.join(destination, image_name)

        if path.isfile(output_path):
            continue

        im = cv2.imread(image_path)
        if im is None:
            print ("Cannot load image {}!".format(image_path))
            continue

        face_aligned, face_not_aligned = get_norm_crop(
            detector, image_name, im, target_size, max_size, image_size, align_mode, offset)

        if face_aligned is not None:
            if not path.exists(path.split(output_path)[0]):
                makedirs(path.split(output_path)[0])

            if len(face_aligned) > 1:
                i = 1
                for face in face_aligned:
                    cv2.imwrite('{}_{}{}'.format(output_path[:-4], i, output_path[-4:]), face_aligned[i - 1])
                    i += 1
            else:
                cv2.imwrite(output_path, face_aligned[0])

            '''
            non_aligned_path = path.join(destination[:-1] + '_not_aligned', image_name)

            if not path.exists(path.split(non_aligned_path)[0]):
                makedirs(path.split(non_aligned_path)[0])

            cv2.imwrite(non_aligned_path, face_not_aligned)
            '''

        else:
            error_path = path.join(destination[:-1] + '_error', image_name)

            if not path.exists(path.split(error_path)[0]):
                makedirs(path.split(error_path)[0])

            cv2.imwrite(error_path, im)
            print ("Cannot find a face in the image {}!".format(image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align Images using RetinaFace')
    parser.add_argument('--img_list', '-i', help='File with an image list.')
    parser.add_argument('--source', '-s', help='Path for the images.')
    parser.add_argument('--dest', '-d', help='Folder to save the cropped faces.')
    # default was 112
    parser.add_argument('--image-size', type=int, default=224, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--det-prefix', type=str, default='../../insightface/models/RetinaFace/R50', help='')
    parser.add_argument('--output', default='./', help='path to save.')
    # default was arcface
    parser.add_argument('--align-mode', default='224', help='align mode.')
    args = parser.parse_args()

    target_size = 1200
    max_size = 1200
    offset = 0.1

    detector = RetinaFace(args.det_prefix, 0, args.gpu, 'net3')

    if not path.exists(args.dest):
        makedirs(args.dest)

    crop_faces(detector, args.img_list, args.source, args.dest,
               target_size, max_size, args.image_size, args.align_mode, offset)
