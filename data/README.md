## Data

This directory contains two subdirectories; the [histology_background](/data/histology_backgrounds/) used for synthetic pre-training, and the [waldos](/data/waldos/), the main dataset which the pre-trained model is fine-tuned on. The images in the [waldos](/data/waldos/) directory are a diverse selection of pages from the "Where's Wally?" books, and the annotations, provided in XML format, identify the location of Waldo in each image.

The histopathological images are sourced from the [Lung and Colon Cancer Histopathological Images Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) available on Kaggle. The waldo dataset is sourced from the [FindWaldo](https://github.com/agnarbjoernstad/FindWaldo) repository by Agnar Martin Bjørnstad.