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

### 上传单张图片并识别
url:localhost:5000/single
key:file
value:单张jpeg、png、webp

输出示例：
[
    [
        "/root/LandingNext_ic/upload/1539248532.26/paper_903.jpg",
        "2:paper"
    ]
]

### 上传多张图片并识别
url:localhost:5000/folder
key:file[]
value:多张jpeg、png、webp

输出示例：
[
    [
        "/root/LandingNext_ic/upload/1539248532.26/paper_888.jpg",
        "2:paper"
    ],
    [
        "/root/LandingNext_ic/upload/1539248532.26/paper_903.jpg",
        "2:paper"
    ]
]

### 上传tfrecord文件并识别
url:localhost:5000/binary
key:file
value:tfrecord文件

输出示例：
[
    [
        "/root/LandingNext_ic/upload/1539586491.5763912/convert/test_0.jpg",
        "3:unknown"
    ],
    [
        "/root/LandingNext_ic/upload/1539586491.5763912/convert/test_1.jpg",
        "3:unknown"
    ]
]

### 保存图片的分类
将识别出来的图片放入save目录下对应的文件夹中

### 增加新的分类并训练
将/datasets/parameters.txt文件中的内容改为分类的总数。

将数据按分类放入input_data文件夹下并将数据转为TFRecord格式：
>python download_and_convert_data.py

执行本命令进行训练：
>python train_image_classifier.py

### 固化模型
模型名称为train文件夹下最新的模型
>python freeze_graph.py --input_checkpoint=./train/model.ckpt
