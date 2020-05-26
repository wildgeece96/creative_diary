今回は自宅PCをサーバーとして使う為の準備をした。  
ただ、完全にサーバー化するわけではなく、Ubuntu ServerとWindowsのデュアルブートというイレギュラーな入れ方をした。  

## Ubuntuのインストール  
Ubuntu 18.04 LTS Serverをインストール。  
以下のサイトを参考にインストールした。  
途中の設定する画面がちょくちょく変わっていたが、基本的にデフォルトの設定のまま進んだ。  

[【Ubuntu 18.04 LTS Server】インストールする | The modern stone age.](https://www.yokoweb.net/2018/05/04/ubuntu-18_04-lts-server-install/)

最後に追加でインストールできるsnap(?)を選択できる画面が最後に現れたが、これは最初のUbuntuのインストール段階で他のパッケージをあらかじめインストールできるというもの。  

dockerを使って機械学習用の環境を構築しようと思っていたのでとりあえずdockerだけ選択した。  

```
$ sudo apt update
$ sudo apt upgrade
$ sudo reboot
```
## nvidiaドライバのインストール  
Ubuntu側にgpuのカードを認識してもらう必要があったので以下のサイトの手順に沿ってNvidiaドライバのインストールを行った。  
[開発メモ その114 Ubuntu 18.04でNvidia Driverをインストールする](https://taktak.jp/2018/05/01/2974)  
ubuntu driversがなかったのでそれをインストールしてから行った。  
```
$ sudo apt install ubuntu-drivers-common
$ ubuntu-drivers devices
(ここに差し込まれているグラフィックボードの情報が出るはず)  

# 自動インストール
$ sudo ubuntu-drivers autoinstall
$ reboot
```
これで`nvidia-smi`を実行して以下のような結果が表示されれば成功。  
```
$ nvidia-smi
Wed Mar 27 14:50:27 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.116                Driver Version: 390.116                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   42C    P5    22W / 250W |      0MiB / 11175MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
### Nvidia Docker のインストール  

https://qiita.com/bohemian916/items/7637b9b0b3494f447c03  

https://qiita.com/legokichi/items/7bf3862569cfca122d73    
