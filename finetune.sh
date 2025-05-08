source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module list
source $HOME/anum/Video2Text/py3.9/bin/activate
cd $HOME/anum/Video2Text/
python finetune.py --dir "/groups/gac50547/RaceCommentary/" --frames 2 --step 2 --n 2000 --hf_dataset "RaceCommentaryEn"

