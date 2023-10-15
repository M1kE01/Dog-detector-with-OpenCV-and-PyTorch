# Object Detector with OpenCV and PyTorch

This project aims to detect objects in images (initially it was just a dog-detector, but now detects 21 different object) using the Faster R-CNN architecture powered by PyTorch. The model is trained on a custom dataset created from the VOC2012 dataset and is capable of detecting 21 different classes.

## Source Dataset

The original dataset utilized for this project is the [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset. You can fetch it using:

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

## Custom Dataset

Based on the VOC2012 dataset, we constructed our own dataset using the `VOCDetectionDataset` class, which parses the images and associated annotations. This dataset was limited in size, and precomputed labels were used for faster fetching during training.

## Classes

Our model is designed to detect the following classes:

0: background  
1: aeroplane  
2: bicycle  
3: bird  
4: boat  
5: bottle  
6: bus  
7: car  
8: cat  
9: chair  
10: cow  
11: diningtable  
12: dog  
13: horse  
14: motorbike  
15: person  
16: pottedplant  
17: sheep  
18: sofa  
19: train  
20: tvmonitor  

## Model

The backbone of our model is the pretrained Faster R-CNN with a ResNet50 FPN architecture available in torchvision. It was then finetuned for our custom dataset with 21 classes. Notably, the model was trained for just 3 epochs, yet it produces satisfactory results owing to the strength of transfer learning.

## Getting Started

To play with the model and witness its capabilities, download [this Jupyter Notebook](https://github.com/M1kE01/Dog-detector-with-OpenCV-and-PyTorch/blob/master/ModelPlayground.ipynb) and run it in Google Colab.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- numpy

---

**Note:** Always ensure to have the necessary dependencies installed and use appropriate runtime (preferably with GPU support) in Google Colab for smoother execution.

---

### Contribution

Feel free to fork this repository and submit pull requests. All contributions are welcome!
