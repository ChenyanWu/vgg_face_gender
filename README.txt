该例子是vgg网络微调完成性别识别的例子：
使用的数据集是LFW人脸数据集，总共有13000张人脸，在lfw中的LFW-gender-folds.txt文件中有人脸名称和label 
使用方法：
	1. 将lfw数据集通过人脸定位的程序进行初始化为224×224的大小,节省训练中再处理的时间。
	2. 修改lfw_Path，test_Path的路径，即LFW-gender-folds.txt，face_list.txt的路径，这两个文件分别存放了LFW数据集图片路径和直播视频处理人脸的路径;
	3. 然后运行程序： $ python train.py
注：其中微调的vggface模型会存放在/data/chenjl文件夹下，vgg_face_50.npy为未训练过的初始vggface模型，vggface_100.npy为迭代100次的训练模型，性别正确率在LFW测试集中达到96%。

