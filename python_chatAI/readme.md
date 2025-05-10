# 本地pycharm运行

```
#启动虚拟环境
conda activate coze
#启动程序
uvicorn app:app --host 0.0.0.0 --port 8080
```

# 远程Windows服务器

1.将文件app.py和config.json移动到桌面目录`桌面/soilKnow/python_chatAI`

2.在该路径下打开终端，安装依赖

```
pip install fastapi uvicorn websockets cozepy
```

3.启动程序

```
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

