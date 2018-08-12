import model_loader
from keras.utils import plot_model

model_names = ['ResNet50', 'VGG19']
stages = ['5', 'final']

for model_name in model_names:
	for stage in stages:
		full_model = model_loader.load_full_model(model_name, random_weights=False, no_cats=2, weight_decay=0, activation='softmax')
		tail_model = model_loader.load_tail_model(model_name, full_model, no_cats=2, weight_decay=0, activation='softmax', stage=stage)
		head_model = model_loader.load_head_model(model_name, random_weights=False, no_cats=2, weight_decay=0, activation='softmax', stage=stage)

		plot_model(full_model, to_file=model_name + '_' + stage + '_' + 'full.png')
		plot_model(tail_model, to_file=model_name + '_' + stage + '_' + 'tail.png')
		plot_model(head_model, to_file=model_name + '_' + stage + '_' + 'head.png')