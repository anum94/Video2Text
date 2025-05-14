source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module list
source $HOME/anum/Video2Text/py3.9/bin/activate
cd $HOME/anum/Video2Text/
python main.py --dir "/groups/gac50547/RaceCommentary" --n 200 --icl True --model_name "llava7b" --step 2 --k 8 --frames 1 --context_window 8192 --wb True --hf_dataset "RaceCommentaryEn"

