# Dockerコンテナ上のMySQLのホストIPアドレスを調べる方法  
コンテナ作成するまでの他の詳しい使い方は以下の記事で紹介している。  

[Dockerで公式のMySQLを使う](https://leck-tech.com/docker/docker-mysql)  

毎回忘れがちになるので自分用のメモ。  
## 手順  
### IDを調べる  
コンテナが作成された前提で話を進める。  
まずは作成したコンテナのIDを調べて起動する。   

```
oharasatsu-no-MacBook-Air-6% docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS                      PORTS                                NAMES
07df1cd7475e        mysql:5.7           "docker-entrypoint.s…"   10 days ago         Exited (255) 45 hours ago   33060/tcp, 0.0.0.0:43306->3306/tcp   mysql
e238f0f9bd65        hello-world         "/hello"                 10 days ago         Exited (0) 4 days ago                                            vigilant_jepsen
```

### 起動  

```
% docker start 07d
```
`docker exec`コマンドを用いて実行。  
```
% docker exec -it 07d /bin/bash
```
### ホストのIPアドレスを調べる  
実行したあとは、以下のようにして調べる。  

```
root@07df1cd7475e:/# cat /etc/hosts
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
ff00::0	ip6-mcastprefix
ff02::1	ip6-allnodes
ff02::2	ip6-allrouters
172.17.0.2	07df1cd7475e
root@07df1cd7475e:/# exit
exit
```
ここの172.17.0.2が求めるIPアドレスとなる。  
これがわかれば、外部アプリからアクセスできる。  
## まとめ  
名称をあらかじめ設定しておけばわかるみたいなことが書いてあったが、自分のコンテナの起動の仕方だと名称を設定できないのでこのやり方で行くしかないようだ。  

## 参考  

[DockerのMySQLコンテナに外部からアクセスする方法まとめ改 - Qiita](https://qiita.com/saken649/items/00e752d89f2a6c5a82f6)
