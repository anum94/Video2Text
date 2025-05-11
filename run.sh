source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module list
source $HOME/anum/Video2Text/py3.9/bin/activate
cd $HOME/anum/Video2Text/
python main.py --dir "/groups/gac50547/RaceCommentary" --n 50 --icl True --step 3 --k 2 --frames 2 --context_window 8192 --wb True --hf_dataset "RaceCommentaryEn"
