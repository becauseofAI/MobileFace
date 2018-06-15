# MobileFace
A face recognition solution on mobile device.

## Performance
| Model | Framework | Size | CPU | LFW | Target |
| :---: |  :---: | :---: | :---: | :---: | :---: |
| MobileFace_Identification_V1 | MXNet | 3.40M | 20ms | - | Actual Scene |
| MobileFace_Identification_V2 | MXNet | 3.41M | 25ms | 99.65% | Benchmark |
## Example
To get fast face feature embedding with MXNet as follow:
```shell
cd example
python get_face_feature_mxnet.py
```

## Visualization
### t-SNE
I used the t-SNE algorithm to visualize in two dimensions the 256-dimensional embedding space. Every color corresponds to a different person(but colors are reused): as you can see, the MobileFace has learned to group those pictures quite tightly. (the distances between clusters are meaningless when using the t-SNE algorithm)  
![t-SNE](./tool/tSNE/tSNE_LFW-100Pair_MobileFace_V1.png "LFW-Aligned-100Pair MobileFace_V1")  
To get the t-SNE feature visualization above as follow:
```shell
cd tool/tSNE
python face2feature.py # get features and lables and save them to txt
python tSNE_feature_visualization.py # load the txt to visualize face feature in 2D with tSNE
```
### ConfusionMatrix
I used the ConfusionMatrix to visualize the 256-dimensional feature similarity heatmap of the LFW-Aligned-100Pair: as you can see, the MobileFace has learned to get higher similarity when calculating the same person's different two face photos. Although the performance of the V1 version is not particularly stunning on LFW Dataset, it does not mean that it does not apply to the actual scene.  
![t-SNE](./tool/ConfusionMatrix/ConfusionMatrix_LFW-100Pair_MobileFace_V1.png "LFW-Aligned-100Pair MobileFace_V1")  
To get the ConfusionMatrix feature similarity heatmap visualization above as follow:
```shell
cd tool/ConfusionMatrix
python ConfusionMatrix_similarity_visualization.py
```
## Tool
### Time
To get inference time of MXNet model as follow:
```shell
cd tool/time
python inference_time_evaluation_mxnet.py
```
### MXNet2Caffe
### Merge_bn

## Benchmark
### LFW
The LFW test dataset (aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112) can be download from [GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view) or [BaiduDrive](https://pan.baidu.com/s/1nxmSCch), and then put it (named lfw.bin) in the directory of ```data/LFW-bin```.  
To get the LFW comparison result as follow:
```shell
cd benchmark/LFW
python lfw_comparison.py
```
### MegaFace

## TODO
- [x] MobileFace_Identification
- [ ] MobileFace_Detection
- [ ] MobileFace_Landmark
- [ ] MobileFace_Align
- [ ] MobileFace_Attribute
- [ ] MobileFace_Pose
- [ ] MobileFace_NCNN
- [ ] MobileFace_FeatherCNN
- [x] Benchmark_LFW
- [ ] Benchmark_MegaFace

## Others
Coming Soon!

## Reference
- [**t-SNE**](http://lvdmaaten.github.io/tsne/ "t-SNE")
- [**InsightFace**](https://github.com/deepinsight/insightface "InsightFace")

