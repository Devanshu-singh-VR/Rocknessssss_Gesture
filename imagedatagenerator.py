from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv

train = ImageDataGenerator(
                           rescale=1/255)

data = train.flow_from_directory(directory='D:',
                                 target_size=(400, 400),
                                 )

a = next(data)[0]
