FILE=$1

if [[ $FILE != "cvusa_lggan" && $FILE != "dayton_lggan" && $FILE != "dayton_ablation_lggan" && $FILE != "sva_lggan"]]; 
	then echo "Available datasets are cvusa_lggan, dayton_lggan, dayton_ablation_lggan, sva_lggan"
	exit 1
fi


echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/datasets/LGGAN/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
