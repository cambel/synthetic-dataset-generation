from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
from randimage import get_random_image
import random
import math
from pathlib import Path

def PIL2cv(image):
    return np.asarray(image)


def cv2PIL(numpy_image):
    # mode = "RGB" if numpy_image.shape[2] == 3 else "RGBA"
    return Image.fromarray(np.uint8(numpy_image))


def generate_background(image_size=(500, 500), random_image_resolution=(100, 100)):
    background = get_random_image(random_image_resolution)
    # Approximate to 0-255 range
    return (cv2.resize(background, image_size) * 255).astype(int)


def show_image(myimage):
    fig, ax = plt.subplots(figsize=[3, 3])
    ax.imshow(myimage, cmap='gray', interpolation='none')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def convert_image(numpy_image, save=False):
    img = Image.fromarray(np.uint8(numpy_image)).convert('RGBA')

    datas = img.getdata()

    newData = []

    threshold = 220

    for item in datas:
        if item[0] >= threshold and item[1] >= threshold and item[2] >= threshold:  # white background
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    if save:
        img.save("./New.png", "PNG")
        print("Successful")

    return img


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * \
        (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def random_overlay(background, foreground, scale_factor=(0.4, 0.8)):
    # Scale the foreground image by a random factor
    rand_scale_factor = random.uniform(scale_factor[0], scale_factor[1])

    img_size = (int(foreground.shape[1]*rand_scale_factor),
                int(foreground.shape[0]*rand_scale_factor))
    foreground = cv2.resize(foreground, img_size)

    # Rotate image
    angle = random.randint(0, 360)
    foreground, new_w, new_h = rotate_image(foreground, angle)


    # Sample a random point to overlay the foreground image
    # Keeping the image within the background image
    min_x_y = 0
    max_x = background.shape[0] - foreground.shape[0]
    max_y = background.shape[1] - foreground.shape[1]
    offset = (random.randint(min_x_y, max_x),
              random.randint(min_x_y, max_y))

    # Compute object center point (x, y)
    x_center = offset[0] + new_w/2
    y_center = offset[1] + new_h/2

    object_pose = (x_center, y_center,
                   img_size[0], img_size[1], math.radians(angle))

    # Overlay image
    # Return CV2 image, the offset of the foreground, and its new size
    return overlay_image(background, foreground, offset), object_pose


def overlay_image(background, foreground, offset=(0, 0)):
    """ background/foreground: numpy array / cv2 image"""
    # Convert to Pillow Image
    if foreground.shape[2] == 3:
        fg = convert_image(foreground)
    else:
        fg = cv2PIL(foreground)
    bg = cv2PIL(background)

    bg.paste(fg, offset, mask=fg)
    return PIL2cv(bg)  # return as numpy array


def convert_rbox2poly(rbox):
    """ Convert top left x,y coordinates and image size to poly format.
        Poly format is the x,y coordinates of each corner
    """
    x, y, w, h, theta = rbox

    center = np.array([x, y])

    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.array([w/2 * Cos, -w/2 * Sin])
    vector2 = np.array([-h/2 * Sin, -h/2 * Cos])

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    res = np.concatenate([point1, point2, point3, point4])
    return np.round(res, 0)


def rotate_image(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.WARP_FILL_OUTLIERS,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255,0))
    return outImg, b_w, b_h


def image_resize(image, max_size=100, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # check to see if height is the largest
    if h > w:
        # calculate the ratio of the height and construct the
        # dimensions
        r = max_size / float(h)
        dim = (int(w * r), max_size)

    # otherwise, the width is the largest
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = max_size / float(w)
        dim = (max_size, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def load_image(image_filename):
    imgfile = Path(image_filename)
    if "png" in imgfile.suffix.lower():
        img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img
    elif "jpg" in imgfile.suffix.lower() or "jpeg" in imgfile.suffix.lower():
        img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError("Unsupported image file extension")
