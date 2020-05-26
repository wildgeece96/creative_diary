# Dockerで公式のMySQLを使う  

前回の記事でDockerのインストールまでを行ったので今回は練習としてDocker上でMySQLを使ってみる。  
以下の記事を参考にしてそれを現在(2019年3月)でもできるのかを確かめたもの。  

- [dockerでmysqlを使う - Qiita](https://qiita.com/astrsk_hori/items/e3d6c237d68be1a6f548)  
- [Dockerで使い捨てのMySQL環境を用意する。事前データを投入して起動する。](https://budougumi0617.github.io/2018/05/20/create-instant-mysql-by-docker/)   
## 手順  
### MySQLのイメージのインストール  
`docker pull`コマンドでできる。  
```zsh
%docker pull mysql:5.7
```
自動的にmysqlのImageがダウンロードできる。  
5.7に指定したのは最新版のver8以降だと認証設定が面倒になってしまうため。  
とりあえず使えれば良いので今回は5.7で使う。  

`docker images`でインストールしたimage一覧が表示される。  
mysqlが入っているのもここで確認できる。  
```zsh
% docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mysql               5.7                 ee7cbd482336        9 days ago          372MB
hello-world         latest              fce289e99eb9        2 months ago        1.84kB
```
### containerの作成  
```zsh
% docker container run -d \
  -e MYSQL_ROOT_PASSWORD=mysql \
  -p 43306:3306 --name mysql mysql:5.7
```
起動しているcontainer一覧を確認する。  
```zsh
% docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                                NAMES
07df1cd7475e        mysql:5.7           "docker-entrypoint.s…"   13 seconds ago      Up 11 seconds       33060/tcp, 0.0.0.0:43306->3306/tcp   mysql  

```
起動しているのが確認できた。  
### mysqlの起動  
先ほどの`-p`オプションでport番号を43306に設定していたのでそれを使う。  
```zsh
% mysql -h 127.0.0.1 --port 43306 -uroot -pmysql
mysql: [Warning] Using a password on the command line interface can be insecure.
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 2
Server version: 5.7.25 MySQL Community Server (GPL)

Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>

```

これでmysqlがDocker上で使える。  
少し使い勝手がわからなくて詰まった部分もあるにはあったが一通り準備を整えられたので、次はmysqlを使ってデータを取り出したりすることをしてみたい。  

## おまけ:起動したcontainerを停止、削除したいとき  
`docker ps`コマンドでcontainerのIDを確認する。  

```zsh
% docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                                NAMES
0000b1ce0dcb        mysql:latest        "docker-entrypoint.s…"   3 seconds ago       Up 1 second         33060/tcp, 0.0.0.0:43306->3306/tcp   mysql
```
ここのCONTAINER IDのところに書かれている値を使って停止と削除を行う  

```zsh
% docker stop 0000b1ce0dcb
% docker rm 0000b1ce0dcb
```

## まとめ  
今回はDocker上にMySQLのImageをダウンロードして実際にcontainerを起動しmysqlを立ち上げるところまでをやった。  
あまり多くないステップ数でここまで来れたので少し驚いている。  
