#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=ImageNet
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-08:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python/3.7
source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/Pytorch_ImaneNet_With_wandb .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/CD-FSL_Datasets .
# cp ~/scratch/dataset/imagenet_object_localization_patched2019.tar.gz .
# cp ~/scratch/dataset/ILSVRC_val.zip .

echo "creating data directories"
date +"%T"
cd Pytorch_ImaneNet_With_wandb
cd data
# tar -xzf $SLURM_TMPDIR/imagenet_object_localization_patched2019.tar.gz
# unzip -q $SLURM_TMPDIR/ILSVRC_val.zip

mkdir ChestX-Ray8 EuroSAT ISIC2018 plant-disease

cd EuroSAT
unzip -q ~/scratch/CD-FSL_Datasets/EuroSAT.zip
cd ..

cd ChestX-Ray8
unzip -q ~/scratch/CD-FSL_Datasets/ChestX-Ray8.zip
mkdir images
find . -type f -name '*.png' -print0 | xargs -0 mv -t images
cd ..

cd ISIC2018
unzip -q ~/scratch/CD-FSL_Datasets/ISIC2018.zip
unzip -q ~/scratch/CD-FSL_Datasets/ISIC2018_GroundTruth.zip
cd ..

cd plant-disease
unzip -q ~/scratch/CD-FSL_Datasets/plant-disease.zip

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd Pytorch_ImaneNet_With_wandb

# python main_wandb_adapt.py --dir ./logs/ImageNet_IN/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0 --epochs 10 --batch-size 256 --workers 8 --resume ./logs/ImageNet/checkpoint_best.pkl

python ImageNet_finetune.py --save_dir ./logs/ImageNet_IN/EuroSAT --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_plus
python ImageNet_finetune.py --save_dir ./logs/ImageNet_IN/CropDisease --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_plus
python ImageNet_finetune.py --save_dir ./logs/ImageNet_IN/ISIC --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_plus
python ImageNet_finetune.py --save_dir ./logs/ImageNet_IN/ChestX --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/ImageNet/checkpoint_best.pkl --freeze_backbone --model resnet18_plus



echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/Pytorch_ImaneNet_With_wandb/logs/ImageNet/ ~/scratch/Pytorch_ImaneNet_With_wandb/logs/
cp -r $SLURM_TMPDIR/Pytorch_ImaneNet_With_wandb/wandb/ ~/scratch/Pytorch_ImaneNet_With_wandb/