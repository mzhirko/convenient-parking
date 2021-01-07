# Convenient Parking

### Description
An application for recognizing free parking spaces. [Mask R-CNN](https://github.com/matterport/Mask_RCNN) (model for object detection and instance segmentation on Keras and TensorFlow) was used in the implementation. _I recommend to read its documentation, download weights file and move it to the project folder_.
No manual segmentation of the parking lot is required to detect available parking spaces, the space will be marked as free after the car mask disappears from there.
Due to certain difficulties when running the program on different devices, it was decided to make a [*__Dockerized version__*](https://github.com/mzhirko/convenient-parking/tree/dockerized-version) (located in a separate branch).

### Demo
![Parking - Animated gif output](demo/output.gif)
I hope i will add the version that works with 2nd TF, usage of TF v.1 causes too much _warnings_.
_That's about all..._