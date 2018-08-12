import keras

import resnet50
import vgg19


def load_full_model(model_name, random_weights=False, no_cats=2, weight_decay=0, activation='softmax'):
	"""
	Loads a model with a randomly initialized last layer

		model_name: ResNet50, VGG19
		random_weights: Random weights or ImageNet pre-training
		no_cats: Number of outputs
		weight decay: L2 weight decay for all layers
		activation: Activation of the final layer (None, softmax, sigmoid)
	"""
	input_tensor=keras.layers.Input(shape=(224, 224, 3))

	if random_weights:
		weights = None
	else:
		weights = 'imagenet'

	if model_name == 'ResNet50':
		full_model = resnet50.ResNet50(weights=weights, input_tensor=input_tensor, weight_decay=weight_decay,
            no_cats=no_cats, activation=activation)
	elif model_name == 'VGG19':
		full_model = vgg19.VGG19(weights=weights, input_tensor=input_tensor, weight_decay=weight_decay,
            no_cats=no_cats, activation=activation)
	else:
		raise ValueError('Invalid model_name')

	return full_model


def load_tail_model(model_name, source_model, no_cats=2, weight_decay=0, activation='softmax', stage='final'):
	"""
	Loads the tail of a model, starting from 'stage'
	Copies the weights from source_model

		model_name: ResNet50, VGG19
		source_model: The model which the weights are going to be copied from
		no_cats: Number of outputs
		weight decay: L2 weight decay for all layers
		activation: Activation of the final layer (None, softmax, sigmoid)
		stage: Which stage the tail starts 
	"""
	if model_name == 'ResNet50':
		no_layers_to_remain = len(source_model.layers) - resnet50.get_no_layers(stage)
		input_shape = source_model.layers[-no_layers_to_remain].input_shape[1:]
		input_tensor = keras.layers.Input(shape = input_shape)

		tail_model = resnet50.ResNet50_tail(input_tensor=input_tensor, weight_decay=weight_decay,
                stage=stage, no_cats=no_cats, activation=activation)
	elif model_name == 'VGG19':
		no_layers_to_remain = len(source_model.layers) - vgg19.get_no_layers(stage)
		input_shape = source_model.layers[-no_layers_to_remain].input_shape[1:]
		input_tensor = keras.layers.Input(shape = input_shape)

		tail_model = vgg19.VGG19_tail(input_tensor=input_tensor, weight_decay=weight_decay,
                stage=stage, no_cats=no_cats, activation=activation)
	else:
		raise ValueError('Invalid model_name')

	for ind_layer in range(1, len(tail_model.layers) + 1):
		tail_model.layers[-ind_layer].set_weights(source_model.layers[-ind_layer].get_weights())

	return tail_model


def load_head_model(model_name, random_weights=False, no_cats=2, weight_decay=0, activation='softmax', stage='final'):
	"""
	Loads the head of a model, up to (excluding) 'stage'

		model_name: ResNet50, VGG19
		random_weights: Random weights or ImageNet pre-training
		no_cats: Number of outputs
		weight decay: L2 weight decay for all layers
		activation: Activation of the final layer (None, softmax, sigmoid)
		stage: Which stage the head ends (excluding)
	"""
	head_model = load_full_model(model_name, random_weights=random_weights, no_cats=no_cats, weight_decay=weight_decay, activation=activation)

	if model_name == 'ResNet50':
		no_layers = resnet50.get_no_layers(stage)
	elif model_name == 'VGG19':
		no_layers = vgg19.get_no_layers(stage)
	else:
		raise ValueError('Invalid model_name')

	while len(head_model.layers) > no_layers:
		head_model.layers.pop()
	head_model = keras.models.Model(head_model.input, head_model.layers[-1].output)

	return head_model


def set_trainable_layers(model, model_name, stage):
	"""
	Sets the layers starting from (and including) 'stage' to be trainable
	"""
	for layer in model.layers:
		layer.trainable = False

	if model_name == 'ResNet50':
		no_trainable_layers = len(model.layers) - resnet50.get_no_layers(stage)
	elif model_name == 'VGG19':
		no_trainable_layers = len(model.layers) - vgg19.get_no_layers(stage)
	else:
		raise ValueError('Invalid model_name')

	for layer in model.layers[-no_trainable_layers:]:
		layer.trainable = True

	return model
