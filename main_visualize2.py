import os

import numpy as np
import keras
import keras.backend as K
from PIL import Image
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from vis.input_modifiers import Jitter
from matplotlib import pyplot as plt

from datasethandler import DatasetHandler
from keras_models import model_loader


def main():
	for domain in domains:
		for model_name in model_names:
			for init_opt in init_opts:
				for preprocess_opt in preprocess_opts:
					for stage in stages:
						log_path = os.path.join(log_path_main, domain, model_name, init_opt, stage)
						if not os.path.exists(log_path):
							os.makedirs(log_path)
						fold = '0'
						K.clear_session()

						if init_opt == 'random':
							model = model_loader.load_full_model(model_name, random_weights = True, no_cats=2)
						elif init_opt == 'ImageNet':
							model = model_loader.load_full_model(model_name, random_weights = False, no_cats=2)
						elif init_opt == 'fine-tuned': # fine-tuned from ImageNet
							model_path = os.path.join('log', domain, model_name, 'ImageNet', preprocess_opt, fold, 'stage_5.h5')
							model = model_loader.load_full_model(model_name, no_cats=2)
							model.load_weights(model_path)
						# keras.utils.plot_model(model, to_file=os.path.join(log_path_main, model_name + '.png'))

						if model_name == 'ResNet50':
							end_layer = ResNet50_layer_names[stage]
						elif model_name == 'VGG19':
							end_layer = VGG19_layer_names[stage]

						ind_layer = utils.find_layer_idx(model, end_layer)
						model.layers[-1].activation = keras.activations.linear
						model = utils.apply_modifications(model)

						#for ind_filter in range(model.layers[ind_layer].output_shape[-1]):
						for ind_filter in range(min(200, model.layers[ind_layer].output_shape[-1])):
							if stage == 'stage_1':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=0, max_iter=200)
							elif stage == 'stage_2':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=1, max_iter=200)
							elif stage == 'stage_3':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=2, max_iter=200)
							elif stage == 'stage_4':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=1, max_iter=200)
							elif stage == 'stage_5':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=0, max_iter=200)
							elif stage == 'stage_final':
								img = visualize_activation(model, ind_layer, filter_indices = ind_filter, input_modifiers=[Jitter()], tv_weight=2, max_iter=400)

							img = Image.fromarray(img, 'RGB')
							img.save(os.path.join(log_path, str(ind_filter) + '.png'), 'PNG')


if __name__ == "__main__":
	"""domains = ['IR', 'VL']
	model_names = ['ResNet50', 'VGG19']
	preprocess_opts = ['mean_subtraction', 'scaling']
	init_opts = ['ImageNet', 'random', 'fine-tuned']
	folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5', 'stage_final']"""

	domains = ['IR']
	model_names = ['ResNet50']
	preprocess_opts = ['mean_subtraction']
	init_opts = ['fine-tuned']
	stages = ['stage_5', 'stage_final']

	ResNet50_layer_names = {'stage_1': 'max_pooling2d_1',
					'stage_2': 'activation_10',
					'stage_3': 'activation_22',
					'stage_4': 'activation_40',
					'stage_5': 'add_16',
					'stage_final': 'fc_final'}

	VGG19_layer_names = {'stage_1': 'block1_pool',
					'stage_2': 'block2_pool',
					'stage_3': 'block3_pool',
					'stage_4': 'block4_pool',
					'stage_5': 'block5_pool',
					'stage_final': 'fc_final'}

	log_path_main = os.path.join('log', 'vis2')
	if not os.path.exists(log_path_main):
		os.makedirs(log_path_main)

	"""os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''"""

	main()