# Ubuntu Server上で動かしたJupyter Notebookをリモートアクセスする  

以下のサイトを参考にしてセットアップを行った。  
セキュリティ問題はあるが、とりあえずリモートアクセスできるようになった。  

[How To Set Up Jupyter Notebook with Python 3 on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-18-04)  

## 環境  
- Ubuntu Server 18.04.02 LTS
## 手順  
### Pythonのセットアップ  
先ずはパッケージのupdateを行う。  
```
$ sudo apt update
```
pipを使ってインストールするのでpipを入れる。  
```
$ sudo apt install python3-pip
```

### Jupyter Notebook用の仮想環境の構築  
virtualenvを使ってPythonの仮想環境を構築する。  
先ずはpipを使ってvirtualenvのインストールを行う。  

```
$ sudo -H pip3 install --upgrade pip
$ sudo -H pip3 install virtualenv
```
プロジェクトのフォルダを作成する。  
```
$ sudo -H pip3 install --upgrade pip
$ sudo -H pip3 install virtualenv
```

仮想環境を作成する。  
```
$ virtualenv my_project_env
```
仮想環境を起動する。  
```
$ . my_project_env/bin/activate
```

### Jupyterのインストール  
仮想環境を起動させた状態でJupyterをインストールする。  
```
(my_project_env)$ pip install jupyter
```

正しく起動するかどうか確かめる。  
```
(myproject_env)$ jupyter notebook
[I 08:10:53.593 NotebookApp] Serving notebooks from local directory: /home/soh/private/jupyter_remote
[I 08:10:53.593 NotebookApp] The Jupyter Notebook is running at:
[I 08:10:53.593 NotebookApp] http://localhost:8888/?token=365781ebaa8c18221c3491b812c23b06a58ee56d27a5968d
[I 08:10:53.593 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 08:10:53.595 NotebookApp] No web browser found: could not locate runnable browser.
[C 08:10:53.595 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///run/user/1000/jupyter/nbserver-22662-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=365781ebaa8c18221c3491b812c23b06a58ee56d27a5968d
[I 08:11:05.272 NotebookApp] 302 GET /?token=365781ebaa8c18221c3491b812c23b06a58ee56d27a5968d (::1) 1.32ms

```
`Ctrl`+`c`を押して停止させる。  
そして、`Ctrl` + `d`を押してログアウトする。  

### sshのトンネリングからJupyterにアクセスする  

手元のlocalhostでのポート番号とサーバー上でのポート番号を対応させるようにしておいた状態でssh越しにサーバにリモートアクセスする。  
```
$ ssh -L 8890:localhost:8888 your_server_username@your_server_ip
```
your_server_usernameでサーバ上のユーザ名を、your_server_ipにサーバのIPアドレス(もしくはドメイン名)を入力して接続する。  
この状態で仮想環境を起動してJupyterを立ち上げる  
```
(my_project_env) $ jupyter notebook
```
この時のリンクの`localhost:8888`という部分を`localhost:8890`に変更してブラウザのURLにコピペすれば画面が見えるようになるはず。  

## まとめ
毎回この設定の仕方を忘れるので備忘録がわりにつけておいた。  
