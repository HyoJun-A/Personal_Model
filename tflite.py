
#%%
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier

#%%
# Load input data specific to an on-device ML app.
data = ImageClassifierDataLoader.from_folder('D:/project/Personal_Model/persnoal/')
train_data, test_data = data.split(0.9)


# Customize the TensorFlow model.
model = image_classifier.create(data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
