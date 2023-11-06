# Traffic-violation-detection
## Custom weights can be downloades from - 
 - Motor-Rider Curriculum Learning - https://drive.google.com/file/d/1HjMEy78Uw8Ro9FCPVATeaTs1vF47BDtl/view?usp=sharing
 - Helmet Detection - https://drive.google.com/file/d/1d0HwLlq1MIZEOw09M3bAgK8RV9Rm6eOv/view?usp=sharing
   
# Custom Trainig Procedure on the data
## *Creating Custom.names file*
To ensure that our YOLOv4 model can accurately identify the 30 different classes of objects in our dataset, we need to save the labels of these objects in a file called **`custom.names`**, which should be saved inside the **'YOLOV4_Custom'** directory. Each line in this file corresponds to one of the object classes in our dataset. In our case, since we have 30 different classes of plant diseases and healthy plants, the 'custom.names' file should contain one line for each of these 30 classes, so that our model can correctly recognize and classify them.

**custom.names**
```
Rider
Motorcycle
Helmet
No-Helmet
```


# **Step 1**
## *Creating Train and Test files*
After uploading and unzipping the dataset, the annotated images should be split into train and test sets with a ratio of **80:20**. The location of the images in the train and test sets should be listed in separate files: **YOLOV4_Custom/train.txt** and **YOLOV4_Custom/test.txt**. Each file row should contain the location of one image in the respective dataset. These files will be used during training to access the images in the correct location.

```
/home2/ur10/yolov4/continual_data/motor_rider_data/0000285_5431.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0004388_7973.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/000388_r_37561.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/000424_r_31294.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0001950_14276.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0001815_8860.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0003726_12986.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/000654_r_37332.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0001649_13972.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0002496_540.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0003345_8557.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0004359_8693.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0003640_7997.jpg
/home2/ur10/yolov4/continual_data/motor_rider_data/0001005_6041.jpg
```

To divide all image files into 2 parts. 80% for train and 20% for test, Upload the *`process.py`* in *`YOLOV4_Custom`* directory

This *`process.py`* script creates the files *`train.txt`* & *`test.txt`* where the *`train.txt`* file has paths to 80% of the images and *`test.txt`* has paths to 20% of the images.

You can download the process.py script from my GitHub.

**Open `process.py` specify the path and then run it.**
```Python
%cd {HOME }
# run process.py ( this creates the train.txt and test.txt files in our darknet/data folder )
!python process.py

```
```Python
# list the contents of data folder to check if the train.txt and test.txt files have been created 
!ls
```


# **Step 2**
## *Creating Configuration file for YOLOv4 model training*
Make a file called `detector.data` in the `YOLOV4_Custom` directory.

```

classes = 30
train = ../train.txt
valid = ../test.txt
names = ../custom.names
backup = ../backup
```

* The classes variable indicates the total number of object classes (in this case, 30).
* train and valid variables point to the text files containing the file paths for the training and validation sets, respectively.
* The names variable points to the file containing the names of the object classes, with one class per line.
* Finally, backup points to the directory where the weights of the model will be saved during training.



# **Step 3**
## *Cloning Directory to use Darknet*
Darknet, an open source neural network framework, will be used to train the detector. Download and create a dark network

```Python
%cd {HOME}
!git clone https://github.com/AlexeyAB/darknet
```

# **Step 4** 
## *Make changes in the `makefile` to enable OPENCV and GPU*

```Python
# change makefile to have GPU and OPENCV enabled
# also set CUDNN, CUDNN_HALF and LIBSO to 1

%cd {HOME}/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```
* The `%cd {HOME}/darknet/` command changes the current directory to the darknet directory in the HOME directory.

* The first command replaces the string OPENCV=0 with OPENCV=1 in the Makefile. This is done to enable OpenCV support in Darknet, which is necessary for some image-related tasks.
* The second command replaces the string GPU=0 with GPU=1 in the Makefile. This is done to enable GPU acceleration in Darknet, which can greatly speed up training and inference.
* The third command replaces the string CUDNN=0 with CUDNN=1 in the Makefile. This is done to enable cuDNN support in Darknet, which is an NVIDIA library that provides faster implementations of neural network operations.
* The fourth command replaces the string CUDNN_HALF=0 with CUDNN_HALF=1 in the Makefile. This is done to enable mixed-precision training in Darknet, which can further speed up training and reduce memory usage.

## *Run `make` command to build darknet*
The `!make` command is a Linux command-line instruction that invokes the make utility to compile and build the Darknet codebase based on the configurations specified in the Makefile. This command reads the Makefile in the current directory and compiles the source code by executing various build commands specified in the Makefile. After the compilation process is complete, the make utility generates an executable binary file that can be used to run various Darknet commands and utilities.

```Python
# build darknet 
!make
```

# **Step 5**
## *Making changes in the yolo Configuration file*

Download the `yolov4-custom.cfg` file from `darknet/cfg` directory, make changes to it, and upload it to the `YOLOV4_Custom` folder on your drive .


**Make the following changes:**

1. `batch=64`  (at line 6)
2. `subdivisions=16`  (at line 7)

3. `width = 416` (has to be multiple of 32, increase height and width will increase accuracy but training speed will slow down).  (at line 8)
4. `height = 416` (has to be multiple of 32).  (at line 9)

5. `max_batches = 60000` (num_classes*2000 but if classes are less then or equal to 3 put `max_batches = 6000`)  (at line 20)

6. `steps = 48000, 54000` (80% of max_batches), (90% of max_batches) (at line 22)
 
7. `classes = 30` (Number of your classes) (at line 970, 1058, 1146)
8. `filters = 105` ( (num_classes + 5) * 3 )  (at line 963, 1051, 1139)

Save the file after making all these changes, and upload it to the `YOLOV4_Custom` folder on your drive .



# **Step 6**
## *Downloading Pre-trained weights*
To train our object detector, we can use the pre-trained weights that have already been trained on a large data sets.
```Python
# changing the current drive to the pre-trained-weights directory to download pretrained weights 
%cd {HOME}/pre-trained-weights

# Download the yolov4 pre-trained weights file
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

# **Step 7**
## *Training the model*
As soon as we have all the necessary files and annotated photographs, we can begin our training.
Up till the loss reaches a predefined level, we can keep training. Weights for the custom detector are initially saved once every 100 iterations until 1,000 iterations, after which they are saved once every 10,000 iterations by default.

We can do detection using the generated weights after the training is finished.

```Python
%cd {HOME}/darknet
!./darknet detector train {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/pre-trained-weights/yolov4.conv.137.1 -dont_show -map
```

## *Continue training from where you left*
Continue training from where you left off, your Model training can be stopped due to multiple reasons, like the notebook time out, notebook craches, due to network issues,  and many more,  so you can start your training from where you left off, by passing the previous trained weights. The weights are saved every 100 iterations as ***yolov4-custom_last.weights*** in the **YOLOV4_Custom/backup** folder on your drive.

```Python
# To start training your custom detector from where you left off(using the weights that were last saved)

%cd {HOME}/darknet
!./darknet detector train {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/backup/yolov4-custom_last.weights  -dont_show -map
```

# 
