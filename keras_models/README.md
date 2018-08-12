I modified some of the models (ResNet50 and VGG19) under [keras/applications](https://github.com/fchollet/keras/tree/master/keras/applications). DenseNet is adapted from [flyyufelix/cnn_finetune](https://github.com/flyyufelix/cnn_finetune).

### Added L2 weight decay
Keras doesn't allow adding regularization to model instances. It has to be added while building the model. This means that loading an ImageNet pre-trained model and fine-tuning it with regularization is not possible. I updated the model building functions to take an L2 weight decay parameter, and apply it to all layers.

### Head models
To get outputs from the intermediate stages of a model, it is common to instantiate a new model. However, I observed that the newly instantiated smaller model does not run faster than the original model. Instead, I implemented a new function that builds the model up to a stage. These models can be used to extract features from images, etc.

### Tail models
To fine-tune last x layers of a model, we freeze the rest of the model, and train it the same way. However, this wastes a lot of time in the forward pass. I implemented a function that builds only the part that will be trained.

### Head-tail synergy
The head and tail models described above saves a lot of time while fine-tuning a model. Say we want to fine-tune the last stage of an ImageNet pre-trained model:
* Create a head model up to the last stage
* Extract the features from the dataset using the head model, write them to the disk if they take up too much space
* Create a tail model that is composed of the last stage
* Train the tail model
* Create a full model, copy all weights of the trained tail model to the full model

This saves A LOT of GPU processing power. Moreover, if the original dataset didn't fit into the memory, but the extracted features did, we can also save a lot of access time (this may be less important if the inputs are read into a queue).
