declare -a domains=("IR" "VL")
declare -a models=("ResNet50" "VGG19")
declare -a init_opts=("random")
declare -a preprocess_opts=("mean_subtraction")

for domain in "${domains[@]}"
do
	for model in "${models[@]}"
	do
		for init in "${init_opts[@]}"
		do
			for preprocess in "${preprocess_opts[@]}"
			do
				for indFold in {0..9}
				do
					python main.py $domain $model $init $preprocess $indFold 0
				done
			done
		done
	done
done