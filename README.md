# CIFAR 10 DNN with Keras

Made for educational purposes to teach myself DCNN. It has about 65-70% accuracy last I checked, but then I also put in the testing images for training. So I do not know the current accuracy. Sorry.

## Usage

```
./get_data.sh
pip install -r requirements.txt
python train.py # This took ~2 hours on my laptop with a NVIDIA GT 940M GPU.
python predict.py /path/to/sample/image.png
```