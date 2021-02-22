# Mask-Detector

A proven way to reduce the spread of the Corona Virus is by the use of a Face Mask. Wearing face masks that covers the nose and mouth would curtail the spread of the virus. Hence, it is made mandatory to wear a mask in public places in most parts of the world. However, the face mask causes some inconveniences, such as difficulty in breathing, fogging of eye glasses and so on. Hence, many people wear it incorrectly, thereby exposing their nose and mouth. This is not a good practise.

It is important for people to wear face masks when they are in public places. By making use of the technology available, this project demonstrates the use of a Mask Detector which could detect people wearing masks. What's more, it can also detect people wearing the mask incorrectly. This can be used in public places, for example, at different sections of a shopping mall or store and so on. The model could detect more than one person and accurately predict whether they are wearing a face mask correctly or not. This information could be used to make them wear it or wear it properly.

On the large scale, proper use of face masks could help the humanity end the pandemic earlier. I hope, this project helps to use AI and Computer Vision for the benefit of the society.

# Hardware
This project uses a [NVIDIA Jetson Nano 2GB module](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/education-projects/) and a [Jelly Comb 1080P HD Webcam](https://www.amazon.de/gp/product/B07ZD39ZWK/ref=ppx_yo_dt_b_asin_title_o07_s00?ie=UTF8&psc=1). 

# Software
The Jetson runs on the [JetPack SDK](https://developer.nvidia.com/embedded/jetpack). This devices uses [TensorRT](https://developer.nvidia.com/tensorrt) to run the machine learning networks on the embedded platform.
This project has been based on the guide [Jetson-Inference](https://github.com/dusty-nv/jetson-inference). The network- [DetectNet](https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/detectnet.py) - was given in this Inference and the training of the model on the custom dataset along with labelling was done by myself.


## Machine Learning Model
The model used for doing the object detection was [Single Shot Detector [SSD-Mobilenet]](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md). A pre-trained model such as SSD is a great way to start as it is much better than starting from scratch. This is called Tramsfer Learning. SSD-Mobilenet is a popular network architecture that uses the [SSD-300](https://arxiv.org/abs/1512.02325) Single-Shot MultiBox Detector with a [Mobilenet](https://arxiv.org/abs/1704.04861) backbone for fast and real-time inference. The [PyTorch](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md) framework was used for the Transfer Learning.

## Data Collection
The data was collected by moving around the subject across the camera's field of view and accuractely marking the bounding boxes. Care was taken to cover the subject tightly. This will make sure the accuracy is good. More than 80 images were taken for each case and various light settings and various positions. This will help the model learn well. The tool to collect data can be found [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md). This tool helps creating a dataset in the Pascal VOC format.

## Training 
The dataset collected was trained on the SSD by specifying the type of dataset, the path of the datset and the model directory. The model was later converted to ONNX model.
Training was done for 30 epochs. The training time was about 2 hours and the classification loss could be seen to reduce and stagnate around 18-20 epochs. 

This ONNX model is to be loaded in detectnet, with few specifications, like the following: 

> detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
            
One needs to specify the model path, labels path, details regaring boxes and then the camera for the live camera stream.
            
# Results
Here are the results of this project. It can detect people those are wearing masks, wearing the mask improperly, or not a wearing mask. The below images show the Mask detector at action. The face is detected and bounded by a box. The algorithm also displays the confidence in the prediction within this box.

1. Person wearing a FFP-2 mask correctly. He is bound by a green box and the model has a confidence of 95.7%
![FFP2](https://user-images.githubusercontent.com/63876751/108638876-b2932600-7491-11eb-9cb8-aabddebb8816.jpg)

2. Person wearing a mask incorrectly. He is bound by a blue box and the model has a confidence of 89.4%
![Improper](https://user-images.githubusercontent.com/63876751/108638877-b45ce980-7491-11eb-9344-b8e614151a9e.jpg)

3. Person wearing no mask. He is bound by a yellow box and the model has a confidence of 96.5%
![No-mask](https://user-images.githubusercontent.com/63876751/108638878-b45ce980-7491-11eb-9da1-c5a4c3e0473d.jpg)

