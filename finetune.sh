source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module list
source $HOME/anum/Video2Text/py3.9/bin/activate
cd $HOME/anum/Video2Text/
python finetune.py --frames 1 --step 2 --n_train 50000 --hf_dataset <path_to_dataset> --context_window 2048 --n_test 200 --use_existing True

