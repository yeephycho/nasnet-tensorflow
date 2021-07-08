# 览笛图像分类引擎

## 安装

docker-ce
nvidia-docker
tensorflow:1.8.0-gpu


将压缩包解压至任意目录$dir(终端输入pwd即可获得当前路径）
>nvidia-docker run -it -v "$dir"/LandingNext_ic:/root/ -p 0.0.0.0:5000:5000 tensorflow/tensorflow:1.8.0-gpu bash

进入目录
>cd ~

启动服务
>python run_server.py


## 功能

### 预测tfrecord文件
将tfrecord文件放入input_data目录并运行：
>python3 predict.py

输出结果分别存储在
`./model_output/processed_files/test_"$model"_processed_files.json`
`./model_output/processed_files/test_"$model"_classify_result.json`
文件中。

### 保存图片的分类
将识别出来的图片放入save目录下对应的文件夹中。

### 增加新的分类并训练
将/datasets/parameters.txt文件中的内容改为分类的总数。

将数据按分类放入input_data文件夹下并将数据转为TFRecord格式：
>python3 download_and_convert_data.py

执行本命令进行训练：
>python3 train_image_classifier.py

### 固化模型
模型名称为train文件夹下最新的模型
>python3 freeze_graph.py --input_checkpoint=./train/model.ckpt

将训练好的模型放入model文件夹下，并在predict.py中第107行修改model名称即可使用。
