cd $1

kaggle datasets download -d dschettler8845/objectnet-1-of-10
unzip -q objectnet-1-of-10 -d objectnet-1-of-10
rm objectnet-1-of-10.zip

kaggle datasets download -d dschettler8845/objectnet-2-of-10
unzip -q objectnet-2-of-10 -d objectnet-2-of-10
rm objectnet-2-of-10.zip