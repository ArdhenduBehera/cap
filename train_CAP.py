'''Imports'''
import numpy as np
import os
from keras import layers, backend as K 
from keras.models import Model, load_model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.applications.xception import Xception, preprocess_input

'''Local Imports'''
from custom_validate_callback import CustomCallback
from image_datagenerator import DirectoryDataGenerator
from loupe_keras import NetRVLAD
from RoiPoolingConv import RoiPoolingConv
from SelfAttention import SelfAttention
from SeqAttention import SeqSelfAttention as SeqSelfAttention
from SeqAttention import SeqWeightedAttention as Attention
from SpectralNormalizationKeras import ConvSN2D
from se import squeeze_excite_block


'''Variables'''
batch_size = 8
checkpoint_freq = 5
dataset_dir = "PATH/TO/DATASET/Aircraft"
epochs = 200
image_size = (224,224)
lstm_units = 128
model_name = "CAP_Xception"
nb_classes = 100
optimizer = SGD(lr=0.0001, momentum=0.99, nesterov=True)
train_dir = "{}/train".format(dataset_dir)
val_dir = "{}/test".format(dataset_dir)
validation_freq = 5

os.environ["CUDA_VISIBLE_DEVICES"]="0"




'''Model Methods'''
#get regions of interest of an image (return all possible bounding boxes when splitting the image into a grid)
def getROIS(resolution=33,gridSize=3, minSize=1):
	
	coordsList = []
	step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
	for column1 in range(0, gridSize + 1):
		for column2 in range(0, gridSize + 1):
			for row1 in range(0, gridSize + 1):
				for row2 in range(0, gridSize + 1):
					
					#get coordinates using grid layout
					x0 = int(column1 * step)
					x1 = int(column2 * step)
					y0 = int(row1 * step)
					y1 = int(row2 * step)
					
					if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
						
						if not (x0==y0==0 and x1==y1==resolution): #ignore full image
							
							#calculate height and width of bounding box
							w = x1 - x0
							h = y1 - y0
							
							coordsList.append([x0, y0, w, h]) #add bounding box to list

	coordsArray = np.array(coordsList)	 #format coordinates as numpy array						

	return coordsArray

def crop(dimension, start, end): #https://github.com/keras-team/keras/issues/890
    #Use this layer for a model that has individual roi bounding box
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)

def squeezefunc(x):
    return K.squeeze(x, axis=1)

'''This is to convert stacked tensor to sequence for LSTM'''
def stackfunc(x):
    return K.stack(x, axis=1) 


'''Model Constants'''
ROIS_resolution = 42
ROIS_grid_size = 3
min_grid_size = 2
pool_size=7
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']


'''Model Construction'''
base_model = Xception(weights='imagenet', input_tensor=layers.Input(shape=(image_size[0],image_size[1],3)), include_top=False)

#last convolution layer
base_out = base_model.output

dims = base_out.shape.as_list()[1:]
feat_dim = dims[2]*pool_size*pool_size
base_channels = dims[2]

x = base_out
x = squeeze_excite_block(x) #Added new

#self-attention
x_f = ConvSN2D(base_channels//8, kernel_size=1, strides=1, padding='same')(x)# [bs, h, w, c']
x_g = ConvSN2D(base_channels//8, kernel_size=1, strides=1, padding='same')(x) # [bs, h, w, c']
x_h = ConvSN2D(base_channels, kernel_size=1, strides=1, padding='same')(x)
x_final = SelfAttention(filters=base_channels)([x, x_f, x_g, x_h])

#x_final = base_out

full_img = layers.Lambda(lambda x: K.tf.image.resize_images(x,size=(ROIS_resolution, ROIS_resolution)), name='Lambda_img_1')(x_final) #Use bilinear upsampling (default tensorflow image resize) to a reasonable size

"""Do the ROIs information and separate them out"""
rois_mat =  getROIS(resolution=ROIS_resolution,gridSize=ROIS_grid_size, minSize=min_grid_size)
num_rois = rois_mat.shape[0]

roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois, rois_mat=rois_mat)(full_img)

jcvs = []
for j in range(num_rois):
    roi_crop = crop(1, j, j+1)(roi_pool)
    lname = 'roi_lambda_'+str(j)
    x = layers.Lambda(squeezefunc, name=lname)(roi_crop)
    x = layers.Reshape((feat_dim,))(x)
    jcvs.append(x)
x = layers.Reshape((feat_dim,))(x_final)
jcvs.append(x)

jcvs = layers.Lambda(stackfunc, name='lambda_stack')(jcvs)
x = SeqSelfAttention(units=32, attention_activation='sigmoid', name='Attention')(jcvs) 

x = layers.TimeDistributed(layers.Reshape((pool_size,pool_size, base_channels)))(x)

x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time'))(x)

lstm = layers.LSTM(lstm_units, return_sequences=True)(x) #experiment with different units 64, 256 etc

y = NetRVLAD(feature_size=128, max_samples=num_rois+1, cluster_size=32, output_dim=nb_classes)(lstm)
y = layers.BatchNormalization(name='batch_norm_last')(y)
y = layers.Activation('softmax', name='final_softmax')(y)

model = Model(base_model.input, y)
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)
model.summary()


''' Training information '''
try:
    os.mkdir("./Metrics")
    os.mkdir("./TrainedModels")
except:
    pass

#Every 50 epochs reduce the learning
def epoch_decay(epoch):
    my_lr = K.eval(model.optimizer.lr)
    if epoch % 50 == 0 and not epoch == 0:
       my_lr = my_lr / 10
    print("EPOCH: ", epoch, "Current LR: ", my_lr)
    return my_lr


basic_schedule = LearningRateScheduler(epoch_decay)

metrics_dir = './Metrics/{}'.format(model_name)
output_model_dir = './TrainedModels/{}'.format(model_name)
csv_logger = CSVLogger(metrics_dir + '(Training).csv')
checkpointer = ModelCheckpoint(filepath = output_model_dir + '.{epoch:02d}.h5', verbose=1, save_weights_only=False, period=checkpoint_freq)

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

train_dg = DirectoryDataGenerator(base_directories=[train_dir], augmentor=True, target_sizes=image_size, preprocessors=preprocess_input, batch_size=batch_size, shuffle=True)
val_dg = DirectoryDataGenerator(base_directories=[val_dir], augmentor=False, target_sizes=image_size, preprocessors=preprocess_input, batch_size=batch_size, shuffle=False)

print("train images: ", nb_train_samples)
print("val images: ", nb_val_samples)

model.fit_generator(train_dg, steps_per_epoch=nb_train_samples // batch_size,  epochs=epochs, callbacks=[checkpointer, csv_logger, CustomCallback(val_dg, validation_freq, metrics_dir)])
