<!-- # Robot harvester: works perfect
![robot](assets/example.jpg "robot")
> **Robot harvester: works perfect**\
> Me Myself, Some Supervisor, Some Other Person
> Paper: https://todo.nl -->

## About
Official implementation of [TomatoWUR](https://data.4tu.nl/datasets/e2c59841-4653-45de-a75e-4994b2766a2f/1)
 dataset: 

**An annotated dataset of tomato plants to quantitatively evaluate segmentation, skeletonisation, and plant trait extraction algorithms for 3D plant phenotyping**


The dataset is related to the paper:
[3D plant segmentation: Comparing a 2D-to-3D segmentation method with state-of-the-art 3D segmentation algorithms](https://www.sciencedirect.com/science/article/pii/S1537511025000832)

## Installation
This software is tested on Python 3.11. To install the dependencies, run:
```
pip install -r requirements.txt
```

## Usage
Make sure to extract and download the dataset, this will be done automatically if path can not be found:
```
python3 wurTomato.py --visualise 0
```
For more examples have a look at the example_notebook.ipynb

Settings are described in config file

<center>
    <p align="center">
        <img src="Resources/3D_tomato_plant.png" height="200" />
        <img src="Resources/3D_tomato_plant_semantic.png" height="200" />
        <img src="Resources/3D_tomato_plant_skeleton.png" height="200" />
    </p>
</center>

## Citation
```
@misc{https://doi.org/10.4121/e2c59841-4653-45de-a75e-4994b2766a2f.v1,
  doi = {10.4121/e2c59841-4653-45de-a75e-4994b2766a2f.v1},
  url = {https://data.4tu.nl/datasets/e2c59841-4653-45de-a75e-4994b2766a2f},
  author = {Marrewijk, van, Bart M. and Xin, Bolai and van Daalen, Tim and van Henten, Eldert and Polder, Gerrit and Kootstra, Gert},
  keywords = {3D plant phenotyping, horticulture, tomato, segmentation, plant architecture, point clouds, RGB images, 3D phenotyping, shape-from-silhouette},
  title = {TomatoWUR: an annotated dataset of 3D tomato plants to quantitatively evaluate segmentation, skeletonisation, and plant trait extraction algorithms for 3D plant phenotyping},
  publisher = {4TU.ResearchData},
  year = {2025},
  copyright = {CC BY-SA 4.0},
}
```

## Related research
[2Dto3D segmentation paper](https://github.com/WUR-ABE/2D-to-3D_segmentation)

## Funding
This research is part of AgrifoodTEF: Test and Experiment Facilities for the Agri-Food Domain (101100622)
