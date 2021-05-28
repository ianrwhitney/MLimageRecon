from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

X = 'deer'
Y = 'geese'

sample_Y_image = './train/Y/geese_01.jpg'

# Create a function that will tweak our image to prevent overfitting

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    rescale=1.0 / 255,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img(sample_Y_image)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

# save to dir has to have the file path and the folder must exist
for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix=Y,
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break
