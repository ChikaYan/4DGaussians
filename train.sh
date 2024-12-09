
SCENE=cabbage3
CONFIG=long2

python train.py -s /home/tw554/4DGaussians/data/china/$SCENE --port 6018 --expname "$SCENE/$CONFIG" --configs arguments/dnerf/$CONFIG.py 

python render.py --model_path "/home/tw554/4DGaussians/output/$SCENE/$CONFIG" --configs arguments/dnerf/$CONFIG.py 

python metrics.py -m "/home/tw554/4DGaussians/output/$SCENE/$CONFIG"