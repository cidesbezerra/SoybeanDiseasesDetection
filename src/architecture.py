from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Dropout, BatchNormalization

from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet

class Architecture:
    def resnet50_tl_ft(image_size):
        base_model = ResNet50(input_shape=image_size, include_top=False, weights="imagenet")
        
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False
        
        x=base_model.output
        #x=BatchNormalization()(x)
        x=GlobalAveragePooling2D()(x)
        x=Dense(512,activation='relu', kernel_initializer='he_uniform')(x) # 2048 we add dense layers so that the model can learn more complex functions and classify for better results.
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#2
        x=Dense(1024,activation='relu', kernel_initializer='he_uniform')(x) # 1024 dense layer 2
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#4
        x=Dense(2048,activation='relu', kernel_initializer='he_uniform')(x) # 512 dense layer 3
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#6
        preds=Dense(4,activation='softmax')(x) #final layer with softmax activation for N classes

        model=Model(base_model.input, preds) #specify the inputs and outputs
        
        return model


    def xception_tl_ft(image_size):
        base_model = Xception(input_shape=image_size, include_top=False, weights="imagenet")

        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False
        """
        #for i, layer in enumerate(model.layers):
        #    print(i, layer.name, "-", layer.trainable)

        # add new classifier layers
        flat1 = Flatten()(base_model.layers[-1].output)
        class1 = Dense(2048, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(4, activation='softmax')(class1)
        #output = Dense(4, activation='softmax')(class1)

        model = Model(base_model.inputs, output)

        """
        x=base_model.output
        #x=BatchNormalization()(x)
        x=GlobalAveragePooling2D()(x)
        x=Dense(512,activation='relu', kernel_initializer='he_uniform')(x) # 2048 we add dense layers so that the model can learn more complex functions and classify for better results.
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#2
        x=Dense(1024,activation='relu', kernel_initializer='he_uniform')(x) # 1024 dense layer 2
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#4
        x=Dense(2048,activation='relu', kernel_initializer='he_uniform')(x) # 512 dense layer 3
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#6
        preds=Dense(4,activation='softmax')(x) #final layer with softmax activation for N classes

        model=Model(base_model.input, preds) #specify the inputs and outputs
        

        return model

    def mobilenet_tl_ft(image_size):

        base_model=MobileNet(input_shape=image_size, weights='imagenet',include_top=False) #imports the MobileNetV2 model and discards the last 1000 neuron layer.
        
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False
        
        x=base_model.output
        #x=BatchNormalization()(x)
        x=GlobalAveragePooling2D()(x)
        x=Dense(512,activation='relu', kernel_initializer='he_uniform')(x) # 2048 we add dense layers so that the model can learn more complex functions and classify for better results.
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#2
        x=Dense(1024,activation='relu', kernel_initializer='he_uniform')(x) # 1024 dense layer 2
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#4
        x=Dense(2048,activation='relu', kernel_initializer='he_uniform')(x) # 512 dense layer 3
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#6
        preds=Dense(4,activation='softmax')(x) #final layer with softmax activation for N classes

        model=Model(base_model.input, preds) #specify the inputs and outputs
        return model

    def inseptionv3_tl_ft(image_size):
        base_model = InceptionV3(input_shape=image_size, include_top=False, weights="imagenet")
        
        #for layer in base_model.layers:#[:143]:
        #    layer.trainable = False # False
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False
        
        x=base_model.output
        #x=BatchNormalization()(x)
        x=GlobalAveragePooling2D()(x)
        x=Dense(512,activation='relu', kernel_initializer='he_uniform')(x) # 2048 we add dense layers so that the model can learn more complex functions and classify for better results.
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#2
        x=Dense(1024,activation='relu', kernel_initializer='he_uniform')(x) # 1024 dense layer 2
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#4
        x=Dense(2048,activation='relu', kernel_initializer='he_uniform')(x) # 512 dense layer 3
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)#6
        preds=Dense(4,activation='softmax')(x) #final layer with softmax activation for N classes

        model=Model(base_model.input, preds) #specify the inputs and outputs
        return model









