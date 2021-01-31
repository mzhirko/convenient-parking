# Convenient Parking
### Steps to run
1. Clone repo & _*cd*_ to it
```Terminal
cd convenient-parking/
```
2. Create venv & activate
```Terminal
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
3. Install requirements
```Terminal
pip3 install -r requirements.txt
```
4. Install weights file from M-RCNN
```Terminal
git clone https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```
5. Run the program
```Terminal
python parking.py
```
6. Don't forget to deactivate venv
Checkout to the [*__Dockerized branch__*](https://github.com/mzhirko/convenient-parking/tree/dockerized-version) if there are any problems with requirements.
### Description
An application for recognizing free parking spaces. [Mask R-CNN](https://github.com/matterport/Mask_RCNN) (model for object detection and instance segmentation on Keras and TensorFlow) was used in the implementation. _I recommend to read its documentation, download weights file and move it to the project folder_.
No manual segmentation of the parking lot is required to detect available parking spaces, the space will be marked as free after the car mask disappears from there.
Due to certain difficulties when running the program on different devices, it was decided to make a [*__Dockerized version__*](https://github.com/mzhirko/convenient-parking/tree/dockerized-version) (located in a separate branch).

### Demo
![Parking - Animated gif output](demo/output.gif)
I hope i will add the version that works with 2nd TF, usage of TF v.1 causes too much _warnings_.
_That's about all..._