import tensorflow as tf
import pathlib

images_classes_path = 'C:/Users/marek/OneDrive/Pulpit/studia/sem6/UM/CustomCoins'


batch_size = 32
img_height = 128
img_width = 128

data_dir = pathlib.Path(images_classes_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

def train_data():

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  return train_ds

def val_data():
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return val_ds



