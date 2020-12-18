FILE=$1

echo "Note: available models are dayton_a2g, dayton_g2a, cvusa, dayton_a2g_64, dayton_g2a_64 and sva"
echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/models/GestureGAN/${FILE}_pretrained.tar.gz
TAR_FILE=./checkpoints/${FILE}_pretrained.tar.gz
TARGET_DIR=./checkpoints/${FILE}_pretrained/

wget -N $URL -O $TAR_FILE

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./checkpoints/
rm $TAR_FILE