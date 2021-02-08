from keras.models import Model
from keras.layers import Dense, Flatten

from keras.applications.resnet50 import ResNet50

class Architecture:
    def resnet50_transfer_learning(image_size):
        model = ResNet50(input_shape=image_size, include_top=False, weights="imagenet")
        
        for layer in model.layers:#[:143]:
            layer.trainable = False
        # Check the freezed was done ok
        #for i, layer in enumerate(model.layers):
        #    print(i, layer.name, "-", layer.trainable)

        # add new classifier layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(2048, activation='relu')(flat1)#kernel_initializer='he_uniform'
        #output = Dense(1, activation='sigmoid')(class1)
        output = Dense(4, activation='softmax')(class1)

        model = Model(model.inputs, output)
        return model
