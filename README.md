# Here is Waldo

In this project, we present a rather novel approach to a problem that has left many sleepless at night: the search for Waldo! We utilize a transfer learning approach to identify Waldo, in complex images from the famous "Where's Wally?" books.

We started this project with a completely different mindset than what we ended up with, if you are interested in the original ideas and what led us to create "Cell Waldo", please read the [Methodology](#methodology) section below. The final model is a regional convolutional neural network (RCNN) that has been pre-trained on histopathological images, and then fine-tuned on an annotated Waldo image dataset.

## Data

This [data](/data/) directory contains two subdirectories; the [histology_background](/data/histology_backgrounds/) used for synthetic pre-training, and the [waldos](/data/waldos/), the main dataset which the pre-trained model is fine-tuned on. The images in the [waldos](/data/waldos/) directory are a diverse selection of pages from the "Where's Wally?" books, and the annotations, provided in XML format, identify the location of Waldo in each image.

The histopathological images are sourced from the [Lung and Colon Cancer Histopathological Images Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) available on Kaggle. The waldo dataset is sourced from the [FindWaldo](https://github.com/agnarbjoernstad/FindWaldo) repository by Agnar Martin Bjørnstad.

## Usage

To try out the model yourself, first clone the project and install the required dependencies:

```bash
git clone https://github.com/SepehrAkbari/here-is-waldo.git
cd here-is-waldo

python -m venv venv
source venv/bin/activate  # Windows: `venv\Scripts\activate`
pip install -r requirements.txt
cd src
```

Next, prepare the synthetic "Waldo" object image, and generate the synthetic training data:

```bash
cd cellwaldo
python cell.py
python dataset.py
cd ../
```

Now, you can process the Waldo dataset, and generate the YOLO configuration:

```bash
cd realwaldo
python dataset.py
python helper_yaml.py
cd ../
```

You can then finally train the YOLOv8 model:

```bash
cd cellwaldo
python train.py
cd ../realwaldo
python train.py
cd ../../
```

We suggest reading through the prepared Jupyter notebook in the [notebook](/notebook/here-is-waldo.ipynb) directory for a more interactive and visual exploration of the model's performance and the training process.

```bash
cd notebook
jupyter notebook here-is-waldo.ipynb
```

## Methodology

Our initial exploration began with a conventional computer vision approach, treating Waldo detection as a face detection problem. Our hypothesis was that Waldo's distinctive facial features, designed for children to identify him, would be robust enough for Haar Cascades. We attempted to fine-tune a pre-trained model and enhance it with Haar Cascades, anticipating that a simplified feature set would effectively classify Waldo from non-Waldo regions. We were wrong. We quickly learned that Waldo's face is neither consistently distinctive nor well-represented by Haar Cascades designed for conventional human faces, especially given the varying orientations and stylized nature of the illustrations. Furthermore, the limited resolution and small annotation box sizes in our dataset compounded these challenges, leading to significant under-fitting and poor model performance.

Moving to a more robust deep learning architecture, we then explored using a traditional CNN approach for direct Waldo/non-Waldo image classification. This revealed a pervasive challenge in the problem: the extreme complexity and diversity of the image backgrounds and Waldo's varying contexts. Moreover, with the  scarcity of well-annotated datasets, a common obstacle given the labor-intensive annotation process, our CNN models consistently suffered from rapid overfitting or a complete failure to learn meaningful features. Attempts at traditional data augmentation (rotations, flips, color manipulation, etc.) introduced noise without significant benefit, and advanced techniques like GANs were not feasible due to the complexity and data limitations. Our classification accuracy at this stage remained around a mere 15-25%.

Recognizing the limitations of classification and traditional feature extraction, we pivoted to Regional-CNNs, specifically an FRCNN, leveraging bounding boxes and annotations as ground truth. This marked a significant improvement. The FRCNN approach was more resilient to the data sparsity, and our minimal augmentations proved sufficient to mitigate overfitting. However, a new challenge emerged: the prevalence of characters visually similar to Waldo in the images. This led to high recall but low precision, as the model struggled to assign a high confidence score uniquely to the true Waldo, making thresholding for successful classification extremely difficult. Despite extensive hyper-parameter tuning and achieving a good confidence, precise detection was not possible. Finally after some very tragic training logs, and a definite misuse of our GPUs, we decided to try a more, creative, approach...

We realized we needed a more innovative solution. Our breakthrough came from adopting a transfer learning strategy, specifically synthetic pre-training. Drawing inspiration from our prior work on a [blood cells analysis pipeline](https://github.com/SepehrAkbari/hemolens), we hypothesized that pre-training a model on a different, yet structurally similar, dataset could provide the foundational feature learning required. We chose a histopathological image dataset, specifically lung and colon cancer images, for its visual complexity but comparatively lower diversity in target objects. We synthetically introduced "Cell Waldos", distinct red-and-white striped objects, into these images and trained a YOLOv8 model from scratch on thousands of these generated examples. This synthetic pre-training yielded near-perfect detection results. We then fine-tuned this pre-trained YOLOv8 model on the real Waldo dataset, strategically freezing most of the layers to retain the learned general features, allowing specialization on low-data targets. This approach significantly improved our confidence metrics for true Waldo detections, achieving a confidence rating of 70-90% on Waldo-containing bounding boxes, demonstrating the efficacy of domain transfer for overcoming severe data limitations in highly challenging visual search tasks.

Looking ahead, significant avenues exist for further enhancing this model's performance. Two key directions include exploring Vision Transformers (ViTs) to leverage their superior spatial reasoning capabilities for handling complex visual scenes, and applying advanced domain adaptation techniques to further bridge the synthetic-to-real data gap. While these represent promising future work, our current creative approach yielded insightful results, leading us to a deeper understanding of the problem's inherent challenges and culminating in the reflections detailed in our [conclusion](#conclusion).

## Conclusion

Our journey through this project highlighted an interesting insight: the inherent nature of a problem dictates the effectiveness of an AI solution. The "Where's Wally?" books are fundamentally designed as a human search puzzle, intentionally engineered to be time-consuming and challenging through deceptive visual cues like similar characters, complex backgrounds, and diverse contexts. Our experimentations, from Haar Cascades to FRCNNs, consistently underscored the difficulty of automating a task that deliberately employs visual ambiguity and distractors. The results achieved through synthetic pre-training and transfer learning, while a relative technical success in object detection under data scarcity, ultimately reinforced the understanding that certain problems are constructed to defy straightforward algorithmic classification. This project served as a critical reminder to align AI approaches with the intrinsic design and purpose of the data, rather than blindly applying models to problems that are, by their very nature, designed for human, nuanced perceptual engagement.

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).