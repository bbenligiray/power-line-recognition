import os

import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.io import savemat
import matplotlib.pyplot as plt
import keras.backend as K

from datasethandler import DatasetHandler
from keras_models import model_loader


def plot_roc():
	for domain in domains:
		for model_name in model_names:
			for init_opt in init_opts:
				for preprocess_opt in preprocess_opts:
					scores = []
					labels = []
					for ind_fold, fold in enumerate(folds):
						K.clear_session()
						dh = DatasetHandler(domain)
						dataset_fold = dh.get_fold(ind_fold, preprocess_opt)

						model_path = os.path.join('log', domain, model_name, init_opt, preprocess_opt, fold, 'stage_5.h5')
						model = model_loader.load_full_model(model_name, no_cats=2)
						model.load_weights(model_path)

						scores.append(model.predict(dataset_fold['test_data'], batch_size=10))
						labels.append(dataset_fold['test_labels'])

					scores = np.concatenate(scores)[:, 1]
					labels = np.concatenate(labels)[:, 1]

					fpr, tpr, _ = roc_curve(labels, scores)
					roc_auc = auc(fpr, tpr)
					savemat(os.path.join('log', 'roc', domain + '_' + model_name + '_' + init_opt + '_' + preprocess_opt + '.mat'), {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})

					plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
					plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
					plt.xlim([0.0, 1.0])
					plt.ylim([0.0, 1.05])
					plt.xlabel('False Positive Rate')
					plt.ylabel('True Positive Rate')
					plt.title('Receiver operating characteristics curve')
					plt.legend(loc="lower right")
					plt.savefig(os.path.join('log', 'roc', domain + '_' + model_name + '_' + init_opt + '_' + preprocess_opt))
					plt.close()


def main():
	file_accs = open(os.path.join('log', 'log_accs.txt'), 'w')
	file_false_images = open(os.path.join('log', 'log_false_images.txt'), 'w')

	for domain in domains:
		for model_name in model_names:
			for init_opt in init_opts:
				for preprocess_opt in preprocess_opts:
					accs_final_layer = []
					accs_stage_5 = []
					false_images_final_layer = []
					false_images_stage_5 = []
					for fold in folds:
						log_path = os.path.join('log', domain, model_name, init_opt, preprocess_opt, fold, 'log.txt')
						with open(os.path.join(log_path)) as f:
							lines = f.readlines()

						ind = lines.index('Final layer\n')
						del lines[ind]
						accs_final_layer.append(float(lines[ind].strip()))

						ind = lines.index('Stage 5\n')
						del lines[ind]
						accs_stage_5.append(float(lines[ind].strip()))

						ind1 = lines.index('Final layer\n')
						ind2 = lines.index('Stage 5\n')

						for ind in range(ind1 + 1, ind2):
							false_images_final_layer.append(lines[ind].strip())
						for ind in range(ind2 + 1, len(lines)):
							false_images_stage_5.append(lines[ind].strip())

					false_images_final_layer.sort()
					false_images_stage_5.sort()

					file_accs.write(domain + ' - ' + model_name + ' - ' + init_opt + ' - ' + preprocess_opt + '\n')
					file_accs.write('Final layer\n')
					acc_final_layer = sum(accs_final_layer) / float(len(accs_final_layer))
					file_accs.write(str(100 - 100 * acc_final_layer) + '\n')
					file_accs.write('Stage 5\n')
					acc_stage_5 = sum(accs_stage_5) / float(len(accs_stage_5))
					file_accs.write(str(100 - 100 * acc_stage_5) + '\n\n')

					file_false_images.write(domain + ' - ' + model_name + ' - ' + init_opt + ' - ' + preprocess_opt + '\n')
					file_false_images.write('Final layer\n')
					file_false_images.write(str(false_images_final_layer) + '\n')
					file_false_images.write('Stage 5\n')
					file_false_images.write(str(false_images_stage_5) + '\n\n')		

	file_accs.close()
	file_false_images.close()
	plot_roc()


if __name__ == "__main__":
	domains = ['IR', 'VL']
	model_names = ['ResNet50', 'VGG19']
	preprocess_opts = ['mean_subtraction', 'scaling']
	init_opts = ['ImageNet', 'random']
	folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	if not os.path.exists('log'):
		os.mkdir('log')

	main()