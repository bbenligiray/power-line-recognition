import os
import argparse

import numpy as np

from datasethandler import DatasetHandler
from keras_models import model_loader
import trainer
from multi_gpu import to_multi_gpu


def main():
	# initialize log file
	file_log = open(os.path.join(log_path, 'log.txt'), 'w')

	file_log.write(domain + ' - ' + str(ind_fold) + '\n')
	file_log.write(model_name + '\n')
	file_log.write(init_method + '\n')
	file_log.write(preprocess_method + '\n')

	# read dataset
	dh = DatasetHandler(domain)
	dataset_fold = dh.get_fold(ind_fold, preprocess_method)

	if init_method == 'random':
		model = model_loader.load_full_model(model_name, random_weights=True, no_cats=2, weight_decay=0.001)
	elif init_method == 'ImageNet':
		model = model_loader.load_full_model(model_name, random_weights=False, no_cats=2, weight_decay=0.001)

	# train the last layer
	accs = []
	false_images = []

	model = model_loader.set_trainable_layers(model, model_name, 'final')
	learning_rate = 0.1
	for ind_iter in range(5):
		model = trainer.train_model(model, dataset_fold, learning_rate)
		learning_rate /= 2
	acc, false_image = trainer.test_model(model, dataset_fold)
	accs.append(acc)
	false_images.append(false_image)
	model.save_weights(os.path.join(log_path, 'final_layer.h5'))

	# fine-tune stage 5 and onwards
	model = model_loader.set_trainable_layers(model, model_name, '5')
	learning_rate = 0.01
	for ind_iter in range(5):
		model = trainer.train_model(model, dataset_fold, learning_rate)
		learning_rate /= 2
	acc, false_image = trainer.test_model(model, dataset_fold)
	accs.append(acc)
	false_images.append(false_image)
	model.save_weights(os.path.join(log_path, 'stage_5.h5'))

	# record accuracies
	file_log.write('Final layer\n')
	file_log.write(str(accs[0]) + '\n')
	file_log.write('Stage 5\n')
	file_log.write(str(accs[1]) + '\n')

	# record falsely classified images
	file_log.write('Final layer\n')
	for fi in false_images[0]:
		file_log.write(fi + '\n')
	file_log.write('Stage 5\n')
	for fi in false_images[1]:
		file_log.write(fi + '\n')

	file_log.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('domain', choices=['IR', 'VL'])
	parser.add_argument('model', choices=['ResNet50', 'VGG19'])
	parser.add_argument('init', choices=['random', 'ImageNet'])
	parser.add_argument('preprocess', choices=['mean_subtraction', 'scaling'])
	parser.add_argument('ind_fold', type=int, choices=range(10))
	parser.add_argument('ind_gpu', type=int, choices=range(4))
	args = parser.parse_args()
	domain = args.domain
	model_name = args.model
	init_method = args.init
	preprocess_method = args.preprocess
	ind_fold = args.ind_fold
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ind_gpu)

	log_path = os.path.join('log', domain, model_name, init_method, preprocess_method, str(ind_fold))
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	main()