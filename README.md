# DL-OBJECT-DETECTION
Object Detection is a Computer Vision task used to identify objects in an image and locate their exact position.
While standard image classification assigns a single label to an entire image, object detection pinpoints multiple distinct entities within the same frame
Basic Process:- 
Input Image → Feature Extraction → Object/Region Detection → Classification + Bounding Box → Final Detection

<img width="1312" height="1199" alt="image" src="https://github.com/user-attachments/assets/5d364835-4167-42c6-8441-f33a1fc5fd3a" />
# Object Detection Frameworks Overview

This document outlines the definitions and standard workflows for two primary object detection libraries: Detectron2 and the TensorFlow Object Detection (TFOD) API.

---

## Detectron2

### Definition
Detectron2 is an open-source, PyTorch-based computer vision library developed by Facebook AI Research (FAIR). It provides state-of-the-art algorithms for object detection, instance segmentation, panoptic segmentation, and keypoint detection. Designed with a highly modular, object-oriented architecture, it allows researchers and engineers to easily build, train, and deploy complex neural networks by overriding default configurations and extending base classes directly in Python.

### Standard Workflow

1. **Environment Setup**
   * Install PyTorch and `torchvision`.
   * Install Detectron2 either via pre-built wheels or by compiling directly from the source repository.

2. **Data Preparation**
   * Detectron2 natively parses datasets formatted in the COCO JSON standard. 
   * For custom annotation formats, you write a Python function that reads your annotations and returns a specific list of dictionaries.
   * Register this dataset to the global catalog using `DatasetCatalog.register("my_dataset", my_custom_function)`.

3. **Model Configuration**
   * Load the default configuration object using `get_cfg()`.
   * Merge this configuration with a base YAML file (e.g., a pre-trained Faster R-CNN architecture).
   * Modify hyperparameters directly in the Python script (e.g., setting `cfg.SOLVER.MAX_ITER`, `cfg.MODEL.ROI_HEADS.NUM_CLASSES`).

4. **Training**
   * Instantiate a trainer object using `DefaultTrainer(cfg)`.
   * Execute the training loop by calling `trainer.train()`. 

5. **Inference and Evaluation**
   * For predictions, create a predictor object via `DefaultPredictor(cfg)` and pass an image array to receive bounding box coordinates and class scores.
   * For formal evaluation, use built-in evaluators like `COCOEvaluator` to compute mean Average Precision (mAP) against a validation dataset.

---

## TensorFlow Object Detection API (TFOD)

### Definition
The TensorFlow Object Detection API (TFOD) is an open-source framework built on top of TensorFlow 2.x by Google. It is designed to construct, train, and deploy object detection models, offering a comprehensive "model zoo" of pre-trained architectures. TFOD is heavily optimized for production environments, making it a standard choice for exporting models to edge devices, Android applications (via TFLite), and web browsers (via TF.js).

### Standard Workflow

1. **Environment Setup**
   * Install TensorFlow 2.x.
   * Compile the required Protocol Buffers (protobuf) libraries.
   * Install the `object_detection` package via the command line.

2. **Data Preparation**
   * Create a `label_map.pbtxt` file that explicitly defines the mapping between class IDs and string names.
   * Convert all raw images and XML/JSON annotations into a strict binary format called **TFRecords**. This is mandatory for TFOD to ingest data efficiently during training.

3. **Model Configuration**
   * Download a pre-trained model and its corresponding `pipeline.config` file.
   * Open the `.config` text file and manually update the file paths to point to your generated TFRecords and label map.
   * Adjust training parameters, such as batch size and learning rate schedules, directly within this file.

4. **Training**
   * Execute the master training script (`model_main_tf2.py`) via the command-line interface.
   * Pass arguments defining the model directory and the path to your customized `pipeline.config`.
   * Monitor the loss and accuracy metrics locally using TensorBoard.

5. **Inference and Export**
   * Run the exporter script (`exporter_main_v2.py`) to freeze the computational graph.
   * Save the output as a TensorFlow `SavedModel`. This self-contained directory can then be deployed for inference or converted into specialized formats for mobile or edge deployment.


   <img width="1312" height="1199" alt="image" src="https://github.com/user-attachments/assets/fcd7e1d9-9382-4891-8852-2f279a9db3e7" />
