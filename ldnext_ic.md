### 上传单张图片并识别
url:172.16.0.228:5000/single
key:file
value:单张jpeg、png

输出示例：
\[
[
    "/root/LandingNext_ic/upload/1539248532.26/paper_903.jpg",
    "2:paper"
]
]

### 上传多张图片并识别
url:172.16.0.228:5000:5000/folder
key:file\[]
value:多张jpeg、png

输出示例：
\[
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
url:172.16.0.228:5000:5000/binary
key:file
value:tfrecord文件

输出示例：
\[
[
    "/root/LandingNext_ic/upload/1539586491.5763912/convert/test_0.jpg",
    "3:unknown"
],
[
    "/root/LandingNext_ic/upload/1539586491.5763912/convert/test_1.jpg",
    "3:unknown"
]
]
