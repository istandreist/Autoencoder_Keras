import cv2
import os
import keras
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
from imutils import paths

os.environ['KMP_DUPLICATE_LIB_OK']='True'

autoencoder = tf.keras.models.load_model('/Users/andregomes/Documents/dev/computerVisionProjects/autoencoderKeras/autoencoder_model.h5')

def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    
    return x



defects = []
defectsImagesPaths = sorted(list(paths.list_images("/Volumes/Gomes/TetraPak curves images/defects/")))
for defectsImagesPath in defectsImagesPaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(defectsImagesPath, 0)
    image = cv2.resize(image, (100, 100))
    image = img_to_array(image)
   
    # extract the class label from the image path and update the
    # labels list

    defects.append(image)
defects = np.expand_dims(defects, axis=-1)
defects = defects.astype("float32") / 255.0

errorsDefects = []
decoded = autoencoder.predict(defects)
outputs = None
# loop over our number of output samples
for i in range(0, 11):
	# grab the original image and reconstructed image
	original = (defects[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")


	mse=np.mean((original-recon)**2)
	errorsDefects.append(mse)


	# stack the original and reconstructed image side-by-side
	output = np.hstack([original, recon])
	
    # if the outputs array is empty, initialize it as the current
	# side-by-side image display
	if outputs is None:
		outputs = output
	# otherwise, vertically stack the outputs
	else:
		outputs = np.vstack([outputs, output])









   
# save the outputs image to disk

print ("Erros defects: " + str(errorsDefects))
cv2.imwrite("output_defects.png", outputs)


