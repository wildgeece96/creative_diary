# DockerのコンテナーのIPアドレスを調べてMySQLコンテナに外部アクセスできるようにする  
[DockerのMySQLコンテナに外部からアクセスする方法まとめ改](https://qiita.com/saken649/items/00e752d89f2a6c5a82f6)  
に載っていた方法。

とりあえず外部からアクセスする際にはホストとなるMySQLコンテナのIPアドレスがないと始まらない。  

まずは`docker ps`で稼働しているコンテナ一覧を取得する。  
```
% docker ps
```
その後、稼働しているMySQLコンテナのIDを確認したら、
以下のコマンドで`docker exec`する。  
```
% docker exec -it 調べたID /bin/bash
```
ここで開かれたところでifconfigを使って調べることができないので、
```
% cat /etc/hosts
root@07df1cd7475e:/# cat /etc/hosts
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
ff00::0	ip6-mcastprefix
ff02::1	ip6-allnodes
ff02::2	ip6-allrouters
172.17.0.3	07df1cd7475e
```
で一番下に書いてある172.17.0.3が今回知りたかったIPアドレス。  
