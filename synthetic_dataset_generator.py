import cv2
import random

import glob
import yaml

from utils import *

from tqdm import tqdm


# load configuration file
# Open the file and load the file
with open('/home/cambel/dev/CUCUMBER-9/sythetic_data_generator/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    # print(config)

# load input files: target images and backgrounds
target_objects = list(config['target_objects'].keys())

# Load files per target object
target_objects_files = {}
for obj in target_objects:
    target_objects_files.update(
        {obj: glob.glob(config['target_object_folder']+"/"+obj+"/*")})

background_files = glob.glob(config['background']['background_folder']+"/*")

file_counter = 1


def save_annotations(output_folder, image, annotations):
    output_dir = Path(output_folder)

    if not output_dir.exists():
        output_dir.mkdir()
    if not output_dir.joinpath("images").exists():
        output_dir.joinpath("images").mkdir()
    if not output_dir.joinpath("labelTxt").exists():
        output_dir.joinpath("labelTxt").mkdir()

    # Generate sequential filename
    global file_counter
    filename = "{:05d}".format(file_counter)
    output_image = output_dir.joinpath("images", filename + ".png")
    output_label = output_dir.joinpath("labelTxt", filename + ".txt")

    # save image
    cv2.imwrite(filename=str(output_image), img=cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

    # save labels
    with open(output_label, mode="w") as f:
        for line in annotations:
            f.write(" ".join(str(item) for item in line) + "\n")
    
    file_counter += 1


def generate_sample(num_of_objs=1, background_type="RANDOM"):
    # Set background
    if background_type == "IMAGE":
        background = load_image(random.choice(background_files))
        background = cv2.resize(background, (config["background"].get("size", 500), config["background"].get("size", 500)))
    elif background_type == "RANDOM":
        background = generate_background()
    else:
        raise ValueError("Invalid background_type %s" % background_type)

    annotations = []

    # Add target objects
    for _ in range(num_of_objs):
        target_object = random.choice(target_objects)
        image_filename = random.choice(target_objects_files[target_object])
        object_image = load_image(image_filename)

        if object_image is None:
            print("ERROR: ", image_filename)

        # resize image
        object_image = image_resize(
            object_image, max_size=config['target_objects'][target_object].get("max_size", 100))

        try:
            background, rbox = random_overlay(
                background, object_image, scale_factor=(0.4, 0.8))
        except Exception as e:
            print("ERROR: ", image_filename)
            print(e)
            raise Exception()
        annotations.append(convert_rbox2poly(
            rbox).tolist() + [target_object, 0])

    save_annotations(config["output_folder"], background, annotations)

# main loop
for _ in tqdm(range(config['target_number_of_samples'])):
    num_of_objs = random.randint(3, config['max_number_of_object_per_image'])
    background_type = random.choice(["RANDOM", "IMAGE", "IMAGE", "IMAGE"])
    generate_sample(num_of_objs, background_type)

