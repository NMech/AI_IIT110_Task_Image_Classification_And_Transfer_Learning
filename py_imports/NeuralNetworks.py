# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from keras.models                  import Sequential
from keras.layers                  import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras              import layers, Model
from tensorflow.keras.callbacks    import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, History, TensorBoard
from messages                      import get_model_summary 
from tensorflow.keras.models       import load_model
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from keras.preprocessing.image     import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras              import initializers
#%%
import logging
from mylogger                      import mylogger
logging_Level = logging.INFO
logger        = mylogger(logging_level = logging_Level)
#%%
class NeuralNetworks:
    
    optimizer_dict = {
            "sgd"     : [tf.keras.optimizers.SGD, 0.01],
            "adam"    : [tf.keras.optimizers.Adam, 0.001],
            "rmsprop" : [tf.keras.optimizers.RMSprop, 0.001],
            "adagrad" : [tf.keras.optimizers.Adagrad, 0.01],
            "adadelta": [tf.keras.optimizers.Adadelta, 1.0],
            "adamax"  : [tf.keras.optimizers.Adamax, 0.002],
            "nadam"   : [tf.keras.optimizers.Nadam, 0.002],
            "ftrl"    : [tf.keras.optimizers.Ftrl, 0.001]}
    
    initializer_dict = {
        "zeros"          : initializers.Zeros(),
        "ones"           : initializers.Ones(),
        "constant"       : initializers.Constant(value=0.5),
        "random_normal"  : initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42),
        "random_uniform" : initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42),
        "glorot_normal"  : initializers.GlorotNormal(seed=42),
        "glorot_uniform" : initializers.GlorotUniform(seed=42),
        "he_normal"      : initializers.HeNormal(seed=42),
        "he_uniform"     : initializers.HeUniform(seed=42),
        "lecun_normal"   : initializers.LecunNormal(seed=42),
        "lecun_uniform": initializers.LecunUniform(seed=42)}
    
    regularizer_dict = {
        "no"    : [None],
        "l1"    : [tf.keras.regularizers.l1, 0.01],
        "l2"    : [tf.keras.regularizers.l2, 0.01],
        "l1_l2" : [tf.keras.regularizers.l1, 0.01, tf.keras.regularizers.l2, 0.01],   
        }
    
    def __init__(self, nclasses, model_selector, input_image_shape, DIRS, img_generator="generator2", **kwargs):
        """
        Keyword arguments:\n
            nclasses          : Number of classes of the classification problem.\n
            model_selector    : string indicating the model architecture to be used.\n
            input_image_shape : tuple of the shape of the input images. (nx_pixels, ny_pixels, n_channels).\n
            DIRS              : list of filemames (full directory path) where the models and the csv log\n
                                files are saved.\n
            img_generator     : Boolean indicating which Image generator is to be used in model training.\n
                                Options:\n
                                    generator1 : Simple generator which does not implement any image transformation.\n
                                    generator2 : Implemented image transformations: Rotation, width and height shifts,\n
                                                 brightness change, shearing, zooming and horizontal flip.\n
            **kwargs          : additional keyword arguments (model hyperparameters such as 'learning_rate').
        """
        self.nclasses = nclasses
        self.model_selector = model_selector
        self.input_image_shape = input_image_shape
        self.mdl_save_dir, self.csv_logger_save_dir, self.history_save_dir, self.architecture_save_dir, self.tensorboard_dir = DIRS
        self.__get_kwargs_vals(**kwargs)
        self.img_generator = img_generator
        self.out_act = "softmax"
        
    def __get_kwargs_vals(self, **kwargs):
        """
        Accessing the additional keyword arguments:\n
        """
        optimizer_get_value    = kwargs.get("model_optim","adam")
        initializer_get_value  = kwargs.get("model_init", "glorot_uniform")
        self.model_optimizer   = NeuralNetworks.optimizer_dict[optimizer_get_value][0]
        self.learning_rate     = float(kwargs.get("learning_rate", NeuralNetworks.optimizer_dict[optimizer_get_value][1]))
        self.model_initializer = NeuralNetworks.initializer_dict[initializer_get_value]
        self.batch_Size        = kwargs.get("batch_size", 32)
        self.nEpochs           = kwargs.get("nEpochs", 10)
        self.model_loss        = kwargs.get("model_loss", "categorical_crossentropy")
        self.model_metrics     = kwargs.get("model_metrics", "accuracy")
        self.patience          = kwargs.get("model_patience", 5)
        self.model_mntr_value  = kwargs.get("model_monitor_value", "val_loss")
        self.nn_act_func       = kwargs.get("nn_act_func", "relu") 
        regularizer_get_value  = kwargs.get("model_regul", "no")
        model_regularizer = NeuralNetworks.regularizer_dict[regularizer_get_value][0]
        if model_regularizer == None:
            self.regularizer = None
        elif model_regularizer == tf.keras.regularizers.l1 or model_regularizer == tf.keras.regularizers.l2:
            self.regularizer = model_regularizer(NeuralNetworks.regularizer_dict[regularizer_get_value][1])
        
        return None
     
    def __save_architecture(self, model):
        """
        Save model architecture in .dat file.\n
        Keyword arguments:\n
            model : tf.keras.Model.\n
        """
        with open(self.architecture_save_dir, "w") as fOut:
            fOut.write(get_model_summary(model))
        return None
    
    def __save_history(self, history):
        """
        Save training history in .csv file.\n
        """
        sep = ";"
        training_loss       = history.history["loss"]
        training_accuracy   = history.history["accuracy"]
        validation_loss     = history.history["val_loss"]
        validation_accuracy = history.history["val_accuracy"]
        if os.path.exists(self.history_save_dir):mode = "a"
        else: mode = "w" 
        
        with open(self.history_save_dir, mode) as file:
            if mode == "w":
                file.write(f"Epoch{sep}Loss{sep}Accuracy{sep}Val Loss{sep}Val Accuracy\n")
            for epoch in range(len(training_loss)):
                file.write(f"{epoch+1}{sep}{training_loss[epoch]}{sep}{training_accuracy[epoch]}{sep}" +
                           f"{validation_loss[epoch]}{sep}{validation_accuracy[epoch]}\n")
        return None
    
    def __select_model(self, mdl_name, unfreeze_layers_original=0, add_cnn_on_top=True, add_deep_on_top=True):
        """
        Select model.\n
        Keyword arguments:\n
            mdl_name        : Name of the neural network model.\n 
            unfreeze_layers : interger indicating the number of layers of the pretrained model\n
                              to be unfrozen and become trainable.\n
            add_cnn_on_top  : Boolean indicating whether the layers defined in __add_cnn_ONTOP\n
                              will be added on the model (based on pretrained models). Always True in this implementation.\n
            add_deep_on_top : Boolean indicating whether the layers defined in __add_deep_ONTOP\n
                              will be added on the model (based on pretrained models). True if the model is the base model\n
                              and False if it is the model used in transfer learning.\n
        """
        if self.model_selector == "C15D3net":
            model = self._C15D3net(mdl_name)  
        elif self.model_selector == "VGG16mod":
            model = self._VGG16mod(mdl_name,  unfreeze_layers_original, add_cnn_on_top, add_deep_on_top) 
        elif self.model_selector == "VGG19mod":
            model = self._VGG19mod(mdl_name, unfreeze_layers_original, add_cnn_on_top, add_deep_on_top)
        elif self.model_selector == "ResNet50mod":
            model = self._ResNet50mod(mdl_name, unfreeze_layers_original, add_cnn_on_top, add_deep_on_top)     

        return model
    
    def __image_data_generator(self, XY):
        """
        Auxiliary function used for generating on the fly augmented images for\n
        the train and validation dataset.\n
        """
        x_train, y_train, x_val, y_val = XY
        if self.img_generator == "generator1": # Nothing to do here. Used for producing datagen
            datagen = ImageDataGenerator(dtype="int8", fill_mode="nearest")
        elif self.img_generator == "generator2":
            datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.5, 1.0], 
                                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        elif self.img_generator == "generator3":
            datagen = ImageDataGenerator(dtype="int8", featurewise_center=True, featurewise_std_normalization=True, 
                                         width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.5, 1.0], 
                                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        datagen.fit(x_train)
        trainGen = datagen.flow(x_train, y_train, batch_size=self.batch_Size)
        validGen = datagen.flow(x_val, y_val, batch_size=self.batch_Size)
        steps_per_epoch  = len(x_train) // self.batch_Size 
        validation_steps = len(x_val) // self.batch_Size
        
        return datagen, trainGen, validGen, steps_per_epoch, validation_steps

    def __compile_model(self, XY, model):
        """
        Compile model procedure.\n
        Keyword arguments:\n
            XY    : tuple of the following format (x_train, y_train, x_val, y_val).\n
            model : tf.keras.Model.\n
        """ 
        es, mc, cl, lr, hs, ts = self.__models_callabacks()

        model.compile(loss=self.model_loss, optimizer=self.model_optimizer(learning_rate=self.learning_rate), metrics=[self.model_metrics])
        datagen, trainGen, validGen, steps_per_epoch, validation_steps = self.__image_data_generator(XY)
        model.fit(trainGen, steps_per_epoch = steps_per_epoch, epochs = self.nEpochs,
                  validation_data = validGen, validation_steps = validation_steps, callbacks = [es, mc, cl, lr, hs, ts])
        self.__save_history(hs)

        return model, datagen
    
    def __load_model(self, XY, model):
        """
        Load model procedure.\n
        Keyword arguments:\n
            XY    : tuple of the following format (x_train, y_train, x_val, y_val).\n
            model : tf.keras.Model.\n
        """
        es, mc, cl, lr, hs, ts = self.__models_callabacks()
        model = load_model(self.mdl_save_dir)
        datagen, trainGen, validGen, steps_per_epoch, validation_steps = self.__image_data_generator(XY)
        model.fit(trainGen, steps_per_epoch = steps_per_epoch, epochs = self.nEpochs,
                  validation_data = validGen, validation_steps = validation_steps, callbacks = [es, mc, cl, lr, hs, ts])
        self.__save_history(hs)
        
        return model, datagen

    def __base_model_selector(self, mdl_name,  mdl_base_name, unfreeze_layers_original, unfreeze_layers_transfer):
        """
        Function used for selecting the base model for the transfer learning procedure.\n
        Keyword arguments:\n
            mdl_name       : Name of the neural network model.\n 
            mdl_base_name  : Filename of the saved base model used in transfer learning.\n
        """
        if self.model_selector == "C15D3net":
            base_model = self.__select_model(mdl_name)
            base_model.load_weights(mdl_base_name, by_name=True, skip_mismatch=True)
            base_output = base_model.layers[-2].output # Exclude the last Dense layer from the base model
            self.__assign_trainable_layers(base_model, unfreeze_layers_transfer+1)
            
        elif self.model_selector in ["VGG16mod", "VGG19mod", "ResNet50mod"]:
            base_model = self.__select_model(mdl_name, unfreeze_layers_original, add_cnn_on_top=True, add_deep_on_top=False) 
            ## !!!!!! unfreeze_layers_original plays no role . Necessary for compilation of model !!!!!
            base_model.load_weights(mdl_base_name, by_name=True, skip_mismatch=True)
            base_output = base_model.layers[-2].output # Set as output layer the last convolution layer.
            self.__assign_trainable_layers(base_model, unfreeze_layers_transfer+1)# +1 is used because output dense layer is present in the architecture and eventually it will be replaced by new output  

        return base_model, base_output
    
    def __models_callabacks(self):
        """
        Function used for defining the callbacks that are to be used in model training.\n
        """
        es = EarlyStopping(monitor=self.model_mntr_value, patience=self.patience, verbose=1) # Define the early stopping callback 
        mc = ModelCheckpoint(self.mdl_save_dir, save_best_only=True, save_weights_only=False) # Define the model checkpoint callback
        cl = CSVLogger(self.csv_logger_save_dir, append=True) # Define the CSVLogger callback
        lr = ReduceLROnPlateau(monitor=self.model_mntr_value, factor=0.5, patience=int(self.patience/2), min_lr=self.learning_rate*0.01)
        hs = History()
        ts = TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1)
        
        return es, mc, cl, lr, hs, ts 
    
    def __assign_trainable_layers(self, model, unfreeze_layers=0):
        """
        Function used for allowing layers in architectures that are based on pretrained models\n
        to be trainable.\n
        Keyword arguments:\n
            model           : tf.keras.Model.\n
            unfreeze_layers : interger indicating the number of layers of the pretrained model\n
                              to be unfrozen and become trainable.\n
        """
        if unfreeze_layers == 0:
            model.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                if layer in model.layers[:-unfreeze_layers]: #the last layer is always the output layer which in all case is to be replaced.
                    layer.trainable = False
                    logger.debug(f"NON-Trainable layer name: {i+1:3d} :: {layer.name}")  
                else:
                    layer.trainable = True
                    logger.debug(f"Trainable layer name: {i+1:3d} :: {layer.name}")  
        return None

    def __structural_block(self, model, filter_xx, kernel_size_xx, pool_size_xx, n_cnn, idx_start, arch_name):
        """
        Structural block used in C15D3net custom architecture.\n
        <<< Conv2D -> BatchNormalization -> >>> * n_cnn times -> MaxPooling2D.\n
        Keyword arguments:\n
            model          : tf.keras.Model.\n 
            filter_xx      : int-> Number of output filters in the convolution.\n
            kernel_size_xx : int-> Height & width of the 2D convolution window.\n
            n_cnn          : int-> Number of Conv2D -> BatchNormalization -> layers of the structural block.\n
            idx_start      : int-> Local index used in naming layers.\n
            arch_name      : str-> Name used in naming layers.\n
        Returns model.
        """
        for i in range(n_cnn):
            model.add(Conv2D(filter_xx, kernel_size_xx, activation = self.nn_act_func, padding="same", name=f"{arch_name}_conv_{idx_start+i:02d}", kernel_initializer=self.model_initializer))
            model.add(BatchNormalization(name=f"{arch_name}_batch_{idx_start+i:02d}"))
        model.add(MaxPooling2D(pool_size = pool_size_xx, name=f"{arch_name}_max_{idx_start+i:02d}"))    
        
        return model
    
    def __add_cnn_ONTOP(self, model):
        """
        Function used for adding additional convolutional layers in architectures that are based on pretrained models.\n
        """
        filter_1, filter_2 = 512, 512
        kernel_Size = 3
        pool_Size   = 2 

        model.add(Conv2D(filter_1, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_1"))
        model.add(BatchNormalization(name="batch_extra_1"))
        model.add(Conv2D(filter_1, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_2"))
        model.add(BatchNormalization(name="batch_extra_2"))
        model.add(Conv2D(filter_1, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_3"))
        model.add(BatchNormalization(name="batch_extra_3"))
        model.add(MaxPooling2D(pool_size = pool_Size, name="maxpool_extra_1"))
        model.add(Dropout(0.50, name="drop_extra_1"))
        model.add(Conv2D(filter_2, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_4"))
        model.add(BatchNormalization(name="batch_extra_4"))
        model.add(Conv2D(filter_2, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_5"))
        model.add(BatchNormalization(name="batch_extra_5"))
        model.add(Conv2D(filter_2, kernel_Size, activation = self.nn_act_func, padding="same", name="conv_extra_6"))
        model.add(BatchNormalization(name="batch_extra_6"))
        return None

    def __add_deep_ONTOP(self, model):
        """
        Function used for adding additional deep fully connected layers in architectures that are based on pretrained models.\n 
        """
        dense_neurons = self.input_image_shape[0]
        model.add(Flatten(name="flatten_extra"))
        model.add(Dense(dense_neurons, activation=self.nn_act_func, name="dense_1_extra", kernel_regularizer=self.regularizer))
        model.add(Dense(dense_neurons, activation=self.nn_act_func, name="dense_2_extra", kernel_regularizer=self.regularizer))
        model.add(Dense(dense_neurons/2, activation=self.nn_act_func, name="dense_3_extra", kernel_regularizer=self.regularizer))

        return None
    
    def _C15D3net(self, mdl_name):
        """
        Custom architecture.\n
        Keyword arguments:\n
            mdl_name : Name of the neural network model.\n   
        """
        arch_name = "C15D3net"
        nx = self.input_image_shape[0]
        filters = [nx/8, nx/8, nx/4, nx/2, nx, 2*nx]
        neurons = [8*nx, 4*nx, 4*nx]
        pool_Size     = 2
        kernel_Size_0 = 7
        kernel_Size_1 = 3

        model = Sequential(name=mdl_name)
        model.add(Conv2D(filters[0], kernel_Size_0, activation = self.nn_act_func, padding="same", input_shape=self.input_image_shape, name=f"{arch_name}_conv_01", kernel_initializer=self.model_initializer))
        model.add(BatchNormalization(name=f"{arch_name}_batch_01"))        
        model = self.__structural_block(model, filters[0], kernel_Size_0, pool_Size, 1, 2, arch_name)
        model = self.__structural_block(model, filters[1], kernel_Size_1, pool_Size, 2, 3, arch_name)
        model = self.__structural_block(model, filters[2], kernel_Size_1, pool_Size, 2, 5, arch_name)
        model = self.__structural_block(model, filters[3], kernel_Size_1, pool_Size, 3, 7, arch_name)
        model = self.__structural_block(model, filters[4], kernel_Size_1, pool_Size, 3, 10, arch_name)
        model = self.__structural_block(model, filters[5], kernel_Size_1, pool_Size, 3, 13, arch_name)
        model.add(Flatten(name=f"{arch_name}_flatten_01"))
        model.add(Dense(neurons[0], activation = self.nn_act_func, name=f"{arch_name}_dense_01", kernel_initializer=self.model_initializer))
        model.add(Dense(neurons[1], activation = self.nn_act_func, name=f"{arch_name}_dense_02", kernel_initializer=self.model_initializer))
        model.add(Dense(neurons[2], activation = self.nn_act_func, name=f"{arch_name}_dense_03", kernel_initializer=self.model_initializer))
        model.add(Dense(self.nclasses, activation=self.out_act   , name=f"dense_out_{mdl_name}", kernel_initializer=self.model_initializer))
        
        return model
    
    def _VGG16mod(self, mdl_name,  unfreeze_layers, add_cnn_on_top=True, add_deep_on_top=True):
        """
        Modified VGG16 architecture.\n
        https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918 .\n
        Architecture:\n
        1.input->2.CONV->3.CONV->4.MAXPOOL->5.CONV->6.CONV->7.MAXPOOL->8.CONV->9.CONV->10.CONV
        ->11.MAXPOOL->12.CONV->13.CONV->14.CONV->15.MAXPOOL->16.CONV->17.CONV->18.CONV->19.MAXPOOL
        -> __add_cnn_ONTOP -> __add_deep_ONTOP -> dense_output.\n
        """
        vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(self.input_image_shape[0], self.input_image_shape[1], 3))
        self.__assign_trainable_layers(vgg16, unfreeze_layers)
        model = Sequential(name=mdl_name)
        model.add(vgg16)
        if add_cnn_on_top == True:
            self.__add_cnn_ONTOP(model)
        if add_deep_on_top == True:
            self.__add_deep_ONTOP(model)
        model.add(Dense(self.nclasses, activation=self.out_act, name=f"vgg16_mod_out_{mdl_name}"))
        
        return model
    
    def _VGG19mod(self, mdl_name,  unfreeze_layers, add_cnn_on_top=True, add_deep_on_top=True):
        """
        Modified VGG19 architecture.\n
        Architecture:\n
        1.input->2.CONV->3.CONV->4.MAXPOOL->5.CONV->6.CONV->7.MAXPOOL->8.CONV->9.CONV->10.CONV->11.CONV->\n
        12.MAXPOOL->13.CONV->14.CONV->15.CONV->16.CONV->17.MAXPOOL->18.CONV->19.CONV->20.CONV->21.CONV->\n
        22.MAXPOOL-> __add_cnn_ONTOP -> __add_deep_ONTOP -> dense_output.\n
        """
        vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=(self.input_image_shape[0], self.input_image_shape[1], 3))
        self.__assign_trainable_layers(vgg19, unfreeze_layers)
        model = Sequential(name=mdl_name)
        model.add(vgg19)
        if add_cnn_on_top == True:
            self.__add_cnn_ONTOP(model)
        if add_deep_on_top == True:
            self.__add_deep_ONTOP(model)
        model.add(Dense(self.nclasses, activation=self.out_act, name=f"vgg19_mod_out_{mdl_name}", kernel_regularizer=l2(0.001)))
        
        return model

    def _ResNet50mod(self, mdl_name, unfreeze_layers, add_cnn_on_top=True, add_deep_on_top=True):
        """
        Modified ResNet50 architecture.\n
        Architecture:\n
        ResNet50 (top layers NOT inclued) -> __add_cnn_ONTOP -> __add_deep_ONTOP -> dense_output.\n
        """
        resnet50 = ResNet50(weights="imagenet", include_top=False, input_shape=(self.input_image_shape[0], self.input_image_shape[1], 3))
        self.__assign_trainable_layers(resnet50, unfreeze_layers)
        model = Sequential(name=mdl_name)
        model.add(resnet50)
        if add_cnn_on_top == True:
            self.__add_cnn_ONTOP(model)
        if add_deep_on_top == True:
            self.__add_deep_ONTOP(model)
        model.add(Dense(self.nclasses, activation=self.out_act, name=f"resnet_mod_out_{mdl_name}")) # this is necessary for not triggering exception!!!!
        return model
      
    def transfer_architecture(self, base_model, base_output, mdl_name):
        """
        This architecture is being used for training a dataset using transfer learning.\n
        Note: Functional API syntax is being used.\n
        In this architecture all layers of the base model are being used except the dense layers\n
        at the top of the base model architecture.\n
        Keyword arguments:\n
            base_model  : The base model to be used during transfer learning.\n
            base_output : Output of the base model.\n
            mdl_name    : Name of the neural network.\n
        Returns model.
        """
        neurons_dense_1 = self.input_image_shape[0]
        neurons_dense_2 = self.input_image_shape[0]/2
        
        flatten_layer    = layers.Flatten(name="flatten_transf")(base_output)
        dense_layer_1    = layers.Dense(neurons_dense_1, activation=self.nn_act_func, name="dense_transf_1")(flatten_layer)
        dense_layer_2    = layers.Dense(neurons_dense_2, activation=self.nn_act_func, name="dense_transf_2")(dense_layer_1)
        prediction_layer = layers.Dense(self.nclasses, activation=self.out_act, name="dense_transf_out")(dense_layer_2)  # Binary classification  
        model_new = Model(inputs=base_model.input, outputs=prediction_layer, name=mdl_name)
        
        return model_new  
    
    def train_model(self, XY, mdl_name, compile_OR_load, unfreeze_layers_original=0):
        """
        Method used in the training procedure of the first dataset.\n
        Keyword arguments:\n
            XY                  : tuple of the following format (x_train, y_train, x_val, y_val).\n
            mdl_name            : Name of the neural network.\n
            compile_OR_load     : Options: "compile" or "load". The first option is\n
                                  being used in case that the model is being trained from start\n
                                  while the second when it is continuing from its saving point.\n
            unfreeze_layers     : interger indicating the number of layers of the pretrained model\n
                                  to be unfrozen and become trainable.\n
        Returns a tf.keras.Model.\n
        """
        model = self.__select_model(mdl_name, unfreeze_layers_original, add_cnn_on_top=True, add_deep_on_top=True)
        if compile_OR_load == "compile":
            model, datagen = self.__compile_model(XY, model)
        elif compile_OR_load == "load":
            model, datagen = self.__load_model(XY, model)
        
        self.__save_architecture(model)
        return model, datagen
    
    def train_transfer_learning_model(self, XY, mdl_name, compile_OR_load, mdl_base_name, unfreeze_layers_original, unfreeze_layers_transfer):
        """
        Method used in the transfer learning procedure.\n
        Keyword arguments:\n
            XY                  : tuple of the following format (x_train, y_train, x_val, y_val).\n
            mdl_name            : Name of the neural network.\n
            compile_OR_load     : Options: "compile" or "load". The first option is\n
                                  being used in case that the model is being trained from start\n
                                  while the second when it is continuing from its saving point.\n
            mdl_base_name       : Filename of the saved base model used in transfer learning.\n
            unfreeze_layers_original     : interger indicating the number of layers of the pretrained model\n
            unfreeze_layers_transfer     :
                                  to be unfrozen and become trainable.\n
        Returns the model object.
        """
        base_model, base_output = self.__base_model_selector(mdl_name, mdl_base_name, unfreeze_layers_original, unfreeze_layers_transfer)
        model = self.transfer_architecture(base_model, base_output, mdl_name) 
        if compile_OR_load == "compile":
            model, datagen = self.__compile_model(XY, model)
        elif compile_OR_load == "load":
            model, datagen = self.__load_model(XY, model)
         
        self.__save_architecture(model)
        return model, datagen