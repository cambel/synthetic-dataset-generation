# synthetic-dataset-generation
Creating synthetic dataset for object classification with rotated bounding boxes

# Instructions
## 1. Preprocessing
Prepare target objects:

1. Collect multiple images of the target objects. Ideally if they are fairly isolated.

<img src="https://github.com/cambel/synthetic-dataset-generation/blob/main/docs/cucumber.jpg?raw=true" alt="c1" width="100"/>

2. Isolate the target object from the background. To remove the background, you could use any tools such as [rembg](https://github.com/danielgatis/rembg). If the image is JPG/JPEG, make sure to make the background flat white, it will be automatically removed during generation of the dataset.

<img src="https://github.com/cambel/synthetic-dataset-generation/blob/main/docs/rembg_cucumber.png?raw=true" alt="c2" width="100"/>

3. Crop to the size of the object. This will be used as the bounding box when generating the synthetic dataset.

<img src="https://github.com/cambel/synthetic-dataset-generation/blob/main/docs/postprocessing_cucumber.png?raw=true" alt="c3" width="100"/>

## 2. Configuring dataset generator
Set the configuration parameters in the `config.yaml`. Directory containing the images that will be used as background as well as the target objects images from the prior step and more.

## 3. Generate dataset
Execute the script
```shell
python synthetic_dataset_generator.py
```