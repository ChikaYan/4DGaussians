
REGION=china
VID_NAME=cabbage3

DATA_DIR=./data/$REGION/$VID_NAME

python vid2frames.py --outdir $DATA_DIR --vid_path /home/tw554/4DGaussians/data/raw_vids/$REGION/$VID_NAME.mp4


colmap feature_extractor --database_path $DATA_DIR/database.db --image_path $DATA_DIR/images  \
    --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 \
    --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path $DATA_DIR/database.db
mkdir $DATA_DIR/sparse
colmap mapper --database_path $DATA_DIR/database.db --image_path $DATA_DIR/images --output_path $DATA_DIR/sparse --Mapper.ba_global_function_tolerance=0.000001


git clone https://github.com/Fyusion/LLFF.git
python LLFF/imgs2poses.py $DATA_DIR/

# cp $DATA_DIR/poses_bounds.npy ./data/multipleview/$workdir/poses_bounds_multipleview.npy




