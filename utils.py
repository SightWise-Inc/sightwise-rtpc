import math
import numpy as np
import cv2
import ffmpeg

# Parameters
MIN_DIST = 200

# Distance to rectangle
def distance(rect, point):
    dx = max(min(rect[0][0], rect[1][0]) - point[0], 0, point[0] - max(rect[0][0], rect[1][0]))
    dy = max(min(rect[0][1], rect[1][1]) - point[1], 0, point[1] - max(rect[0][1], rect[1][1]))
    return math.sqrt(dx*dx + dy*dy)


# Selection algorihtm
select_history = None
def select(image, objects=None, texts=None, target=None):
    global select_history
    # handle temporality
    # TODO
    if target is None: return None
    # if target == "center": target = image.shape[:2] // 2

    closest = None
    min_dist = MIN_DIST

    if texts:
        for idx, text in enumerate(texts): 
            # print(text) # DEBUG
            # extract points
            pt1 = [int(n) for n in text[0][0]]
            pt2 = [int(n) for n in text[0][2]]
            # select
            dist = distance([pt1, pt2], target)
            # print('Text', text[1], ':', dist) # DEBUG
            if dist < min_dist: 
                # closest = text
                closest = { 'box': [pt1, pt2], 'name': text[1] } # hopefully this works
                min_dist = dist

    return closest


# add mask for selective image manipulation
def add_mask(mask, pt1, pt2):
    mask[pt1[1]:pt2[1],pt1[0]:pt2[0]] = 1
    return mask


def apply_highlight(img, weak_mask, strong_mask):
    # weak -= np.logical_and(weak, strong) # remove overlap # unnecessary because strong overwrites weak

    alpha = 2.4 # Simple contrast control
    beta = 30    # Simple brightness control

    weak = np.clip(img * alpha/1.7 + beta/2, 0, 255).astype(np.uint8)
    strong = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    highlighted = np.where(weak_mask, weak, img)
    highlighted = np.where(strong_mask, strong, highlighted)

    return highlighted


def draw_viz(img, objects=None, texts=None, hands=None, selection=None, cursor=None):

    # initialize highlighting masks
    weak = np.zeros(shape=img.shape)
    strong = np.zeros(shape=img.shape)
    # center = (int(img.shape[1]/2), int(img.shape[0]/2))

    # iterate through each objects 
    # (color them in a consistent, aesthetic manner)

    # if objects: 
    #     drawn = draw(img, objects['boxes'], objects['scores'], objects['class_ids'], objects['indices'])

    if texts:
        # iterate through each text
        for i, r in enumerate(texts): 
            # print(r)
            # points
            pt1 = [int(n) for n in r[0][0]]
            pt2 = [int(n) for n in r[0][2]]
            # add highlight
            try: weak = add_mask(weak, pt1, pt2)
            except Exception: print('highlight error')

    # if hands:
    #     img = visualize_hands(hands, img)

    # selected text/object
    if selection:
        # pt1, pt2 = [int(n) for n in selection[0][0]], [int(n) for n in selection[0][2]]
        pt1, pt2 = [int(n) for n in selection['box'][0]], [int(n) for n in selection['box'][1]]
        pt1, pt2 = tuple(pt1), tuple(pt2) # OpenCV's rectangle doesn't like coordinates given in list
        strong = add_mask(strong, pt1, pt2)
        img = cv2.rectangle(img, pt1, pt2, (255,255,255), 15) # outline

    # selection cursor
    if cursor:
        # cursor = [int(cursor[0]*img.shape[0]), int(cursor[1]*img.shape[1])]
        # cursor = [int(cursor[1]*img.shape[1]), int(cursor[0]*img.shape[0])]
        # cursor = [int(cursor[0]*img.shape[1]), int(cursor[1]*img.shape[0])]
        # print('cursor:', cursor)
        img = cv2.rectangle(img, cursor, cursor, (0,100,0), 20)

    # visualization
    # img = cv2.rectangle(img, pt1, pt2, (255,255,255), 15) # TODO add outline to everything!
    highlighted = apply_highlight(img, weak, strong)

    return highlighted


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

