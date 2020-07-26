## Ref:
https://github.com/experiencor/keras-yolo2

## Requirements
1. keras 2
2. tensorflow
3. imgaug
4. cv2
5. 

After installing all required package, follows the `preprocessing.ipnb` files block by block, where you need to pay attention to specific points to run it successfully.

1. Address of data in 2nd block-cell
2. Place Anchors in the ANCHORS variable, if you want to train on another dataset, otherwise leave that compleet block as it is.
3. Add dataset of level2 if you want, which was commented out in the code. By the way, it didn't improve the result.
4. My code is speciifc for `gpu`, if you want to run on cpu, change parallel model to simple keras fit_generator.
5. Train the algo for three round. Make sure you are saving the weights properly, if you want to re-run code afterwards.
 