from random import shuffle
import pickle

import numpy as np
from PIL import Image

import dataset_pickler


class DatasetHandler:

	def __init__(self, domain):
		# pickle the dataset if it's not pickled already
		dataset_pickler.main()
		self.domain = domain
		# load the dataset
		with open('power_lines.p', 'rb') as f:
			self.dataset = pickle.load(f)
			print self.dataset['dataset_info']
		# generate the fold indices once at the start
		self.indices = self.generate_fold_indices()


	def get_all(self, preprocess_method):
		"""
		Returns the entire dataset for a domain.
		Used for feature extraction.
		"""
		no_images = self.dataset['images'][self.domain].shape[0]
		data = np.empty((no_images, 224, 224, 3,), dtype=np.float32)
		for ind_image in range(no_images):
			# read the image
			image = Image.fromarray(self.dataset['images'][self.domain][ind_image], 'RGB')
			# resize it
			image = image.resize((224, 224), resample=Image.BILINEAR)
			# cast to float
			image_np = np.asarray(image, dtype=np.float32)
			# preprocess
			if preprocess_method == 'scaling':
				image_np /= 255
			elif preprocess_method == 'mean_subtraction':
				image_np -= self.dataset['pixel_mean'][self.domain]
			else:
				raise ValueError('Invalid preprocess_method')
			data[ind_image] = image_np

		dataset_all = {}
		dataset_all['data'] = data
		dataset_all['labels'] = self.dataset['labels'][self.domain].astype(np.float32)
		dataset_all['image_names'] = self.dataset['image_names'][self.domain]

		return dataset_all


	def get_fold(self, ind_fold, preprocess_method):
		"""
		Uses the indices created by generate_fold_indices() to draw a fold from the dataset.
		Also applies preprocessing.
		"""
		dataset_fold = {}
		for data_type in ['train', 'val', 'test']:
			no_examples = self.indices[data_type][ind_fold].shape[0]
			
			# read the data and preprocess it
			data = np.empty((no_examples, 224, 224, 3,), dtype=np.float32)
			labels = np.zeros((no_examples, 2), dtype=np.float32)
			for ind_image in range(no_examples):
				shuffled_ind = self.indices[data_type][ind_fold, ind_image]
				# read the image
				image = Image.fromarray(self.dataset['images'][self.domain][shuffled_ind], 'RGB')
				# resize it
				image = image.resize((224, 224), resample=Image.BILINEAR)
				# cast to float
				image_np = np.asarray(image, dtype=np.float32)
				# preprocess
				if preprocess_method == 'scaling':
					image_np /= 255
				elif preprocess_method == 'mean_subtraction':
					image_np -= self.dataset['pixel_mean'][self.domain]
				else:
					raise ValueError('Invalid preprocess_method')
				data[ind_image] = image_np

				labels[ind_image] = self.dataset['labels'][self.domain][shuffled_ind]

			# read the image names
			image_names = np.ndarray(no_examples, dtype=object)
			for ind_image in range(no_examples):
				shuffled_ind = self.indices[data_type][ind_fold, ind_image]
				image_names[ind_image] = self.dataset['image_names'][self.domain][shuffled_ind]

			# shuffle them together
			inds = np.random.permutation(data.shape[0])
			data = data[inds]
			labels = labels[inds]
			image_names = image_names[inds]

			dataset_fold[data_type + '_data'] = data
			dataset_fold[data_type + '_labels'] = labels
			dataset_fold[data_type + '_image_names'] = image_names

		return dataset_fold


	def generate_fold_indices(self, no_example_per_class=2000, no_class=2, no_folds=10,
					train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
		"""
		Generates indices for k-fold cross validation.
		"""
		assert(test_ratio * no_folds == 1)

		no_examples_per_class = np.sum(self.dataset['labels'][self.domain], axis=0)

		no_train = np.round(no_examples_per_class * train_ratio).astype(np.int)
		no_val = np.round(no_examples_per_class * val_ratio).astype(np.int)
		no_test = (no_examples_per_class - no_train - no_val).astype(np.int)

		indices = {'train': np.empty((no_folds, sum(no_train)), dtype=np.int),
					'val': np.empty((no_folds, sum(no_val)), dtype=np.int),
					'test': np.empty((no_folds, sum(no_test)), dtype=np.int)}

		

		shuffled_inds = [range(no_examples_per_class[0]), no_examples_per_class[0] + range(no_examples_per_class[1])]
		shuffle(shuffled_inds[0])
		shuffle(shuffled_inds[1])

		for ind_fold in range(no_folds):
			# class #1
			copied_shuffled_inds = list(shuffled_inds[0])

			indices['test'][ind_fold, :no_test[0]] = copied_shuffled_inds[ind_fold * no_test[0]:(ind_fold + 1) * no_test[0]]
			del copied_shuffled_inds[ind_fold * no_test[0]:(ind_fold + 1) * no_test[0]]

			shuffle(copied_shuffled_inds)
			indices['train'][ind_fold, :no_train[0]] = copied_shuffled_inds[:no_train[0]]
			indices['val'][ind_fold, :no_val[0]] = copied_shuffled_inds[no_train[0]:]

			# class #2
			copied_shuffled_inds = list(shuffled_inds[1])

			indices['test'][ind_fold, no_test[0]:] = copied_shuffled_inds[ind_fold * no_test[1]:(ind_fold + 1) * no_test[1]]
			del copied_shuffled_inds[ind_fold * no_test[1]:(ind_fold + 1) * no_test[1]]

			shuffle(copied_shuffled_inds)
			indices['train'][ind_fold, no_train[0]:] = copied_shuffled_inds[:no_train[1]]
			indices['val'][ind_fold, no_val[0]:] = copied_shuffled_inds[no_train[1]:]

		return indices