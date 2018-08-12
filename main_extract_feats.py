import os

import keras
import numpy as np
from scipy.io import savemat
from sklearn.decomposition import PCA

from datasethandler import DatasetHandler
from keras_models import model_loader


def main():
	domains = ['IR', 'VL']
	preprocess_methods = ['mean_subtraction', 'scaling']
	init_methods = ['random', 'ImageNet']
	model_names = ['ResNet50', 'VGG19']
	stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5']

	ResNet50_layer_names = ['max_pooling2d_1',
					'activation_10',
					'activation_22',
					'activation_40',
					'activation_49']

	VGG19_layer_names = ['block1_pool',
					'block2_pool',
					'block3_pool',
					'block4_pool',
					'block5_pool']

	out_dict = {}
	sampling_method = 'pca'
	for domain in domains:
		for preprocess_method in preprocess_methods:
			# load dataset
			dh = DatasetHandler(domain)
			dataset_all = dh.get_all(preprocess_method)

			for init_method in init_methods:
				for model_name in model_names:
					for ind_stage, stage in enumerate(stages):
						# load model
						keras.backend.clear_session()
						if init_method == 'random':
							model = model_loader.load_full_model(model_name, random_weights=True, no_cats=2, weight_decay=0.001)
						elif init_method == 'ImageNet':
							model = model_loader.load_full_model(model_name, random_weights=False, no_cats=2, weight_decay=0.001)

						# strip layers
						if model_name == 'ResNet50':
							end_layer = ResNet50_layer_names[ind_stage]
						elif model_name == 'VGG19':
							end_layer = VGG19_layer_names[ind_stage]

						model = keras.models.Model(inputs = model.input, outputs = model.get_layer(end_layer).output)

						feats = model.predict(dataset_all['data'])
						feats = np.reshape(feats,(dataset_all['data'].shape[0], -1))

						if sampling_method == 'pca':
							pca = PCA(1024)
							feats = pca.fit_transform(feats)
						elif sampling_method == 'uniform':
							if feats.shape[1] > 16384:
								sample_indices = np.round(np.linspace(0, feats.shape[1] - 1, 16384))
								feats = feats[:, sample_indices.astype(int)]

						name = domain + '_' + preprocess_method + '_' + init_method + '_' + model_name + '_' + stage
						out_dict[name + '_feats'] = feats
						out_dict[name + '_labels'] = dataset_all['labels']
						
	savemat('feats_' + sampling_method + '.mat', out_dict, do_compression=True)


if __name__ == "__main__":
	log_path = 'log'
	if not os.path.isdir(log_path):
		os.mkdir(log_path)
	main()