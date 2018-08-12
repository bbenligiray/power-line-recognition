import os

import numpy as np
import keras.backend as K
from PIL import Image

from datasethandler import DatasetHandler
from keras_models import model_loader
from deep_viz_keras.guided_backprop import GuidedBackprop


def main():
	for domain in domains:
		for model_name in model_names:
			for init_opt in init_opts:
				for preprocess_opt in preprocess_opts:
					log_path = os.path.join(log_path_main, domain, model_name, init_opt)
					if not os.path.exists(log_path):
						os.makedirs(log_path)
					for ind_fold, fold in enumerate(folds):
						K.clear_session()
						dh = DatasetHandler(domain)
						dataset_fold = dh.get_fold(ind_fold, preprocess_opt)

						model_path = os.path.join('log', domain, model_name, init_opt, preprocess_opt, fold, 'stage_5.h5')
						model = model_loader.load_full_model(model_name, no_cats=2)
						model.load_weights(model_path)
						model.compile(optimizer = 'adam',
									loss = 'categorical_crossentropy',
									metrics = ['accuracy'])
						guided_bprop = GuidedBackprop(model)

						for ind_image, image in enumerate(dataset_fold['test_data']):
							if domain == 'VL':
								image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
								image = image[:, :, np.newaxis]
								image = np.repeat(image, 3, axis=2)

							mask = guided_bprop.get_mask(image)

							mask = np.power(mask, 2)
							mask = np.sum(mask, axis=2)
							mask = np.sqrt(mask)

							mask -= np.min(mask)
							norm_max = np.max(mask)
							if norm_max == 0:
								norm_max = 1
							mask /= norm_max
							mask *= 255

							mask = np.uint8(mask)
							img = Image.fromarray(mask, 'L')
							im_name = dataset_fold['test_image_names'][ind_image].split('.')[0]
							img.save(os.path.join(log_path, im_name + '.png'), 'PNG')


if __name__ == "__main__":
	"""domains = ['IR', 'VL']
	model_names = ['ResNet50', 'VGG19']
	preprocess_opts = ['mean_subtraction', 'scaling']
	init_opts = ['ImageNet', 'random']
	folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']"""

	domains = ['IR', 'VL']
	model_names = ['ResNet50']
	preprocess_opts = ['mean_subtraction']
	init_opts = ['ImageNet']
	folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	log_path_main = os.path.join('log', 'vis')
	if not os.path.exists(log_path_main):
		os.makedirs(log_path_main)

	main()