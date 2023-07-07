# CV based Activity Recognition

This project demonstrates activity recognition using a pre-trained model. It takes a video as input and predicts the activity in each frame of the video.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Usage

To run the project using the command below:

Accessing the video frame
``` python <file name> --model resnet-34_kinetics.onnx --classes Actions.txt --input <video path> --gpu 1 --output output.mp4```

Example cmd
``` python Activity.py --model resnet-34_kinetics.onnx --classes Actions.txt --input videos/soccer.mp4 --gpu 1 --output output.mp4```

Accessing the webcam
```python <file name> --model resnet-34_kinetics.onnx --classes Actions.txt```

Example cmd
```python Activity.py --model resnet-34_kinetics.onnx --classes Actions.txt```



