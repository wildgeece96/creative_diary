# Ubuntu Server 18.04 を外部ネットからアクセスできるようにする  

自宅のデスクトップパソコンにUbuntu Serverをインストールしたので、ローカルだけでなく外部ネットからもアクセスできるようになりたいと思い、色々試行錯誤した時の備忘録。  

とりあえず繋げられるようになることが目標なのでそこを頑張りたい。  
## MACアドレスの確認  
Ubuntu Server側のMACアドレスを把握する必要があるので以下のコマンドを使う。  

```
ifconfig | grep ehter
```  


## ルーターの設定  
### ルーターの設定画面を開く  
自分が使っているルーターはドコモの光ルーター 01と呼ばれるもの。  
設定画面へは http://dhr.tool と入力すれば飛べた。  

ここでIDとパスワードの入力を求められたので  
IDにはadminで、パスワードには Web PWを指定する。  
ホーム画面はこんな感じ。  
![router_setup.png](./router_setup.png)  

### MACアドレスからIPアドレスの割り当てをする  
ここの左上にある「詳細モードに切り替え」をクリック。  
![router_setup_advanced.png](./router_setup_advanced.png)  

そしたら、左側の詳細設定の部分をクリックして展開された部分に「DHCP固定割当設定」が表示されるのでそこをクリック。  
![router_mac_setup.png](./router_mac_setup.png)  
ここに指定したいIPアドレスと先ほど取得したMACアドレスを入力する。  
![router_mac.png](./router_mac.png)
追加されると以下のようにリストが出てくる。  
![router_mac_2.png](./router_mac_2.png)  

### ポート設定  
今度は「ポートマッピング設定」をクリックして設定をいじる。  
![port_setup.png](./port_setup.png)  
ポート番号を80~88の間に設定し、IPアドレスは先ほど割り当てた192.168.10.100を使用する。  

おそらくこれでルータ側の設定ができた。  

## ドメイン付きホスト名の取得  
外部ネットからアクセスするためにはルータのIPアドレスではなく、ドメイン付きホスト名を取得した方が良いらしいのでそれを使う。  

### DDNSへの登録  
[MyDNS.JP](https://www.mydns.jp/)  
を利用してDDNSへの登録を行う。  
ここのリンクへ飛ぶと以下のようなページが表示される。  
![mydns_top.png](./mydns_top.png)  
ここの右上のメニューJOIN USのボタンがあるのでそこをクリック。  
![mydns_joinus_top.png](./mydns_joinus_top.png)  
ここをスクロールすると「登録者情報」を入力する欄があるので、指名、住所、電話番号、メールアドレスを入力する。  
![mydns_joinus_2.png](./mydns_joinus_2.png)  
入力が終わり、項目の確認画面でOKボタンを押すと以下のような画面になるので、メールが届くのをまつ。  
![mydns_joinus_3.png](./mydns_joinus_3.png)
メールには**IDとパスワードが記載されたものが届く**ので、そのIDとパスワードをログイン欄に入力する。  
[MyDNS.JP](https://www.mydns.jp/)  
![mydns_login.png](./mydns_login.png)  
![mydns_login_2.png](./mydns_login_2.png)  


ログインしたら、Welcome AdministratorのところのDOMAIN INFOのところをクリック。  
![domain_info.png](./domain_info.png)   
ドメイン名を指定する。  
![domain_info_2.png](./domain_info_2.png)  
選択肢として"mydns.jp~deamon.asia"があるらしく、これのうちのどれか1つ好きなのを選んで、好きなドメイン名を設定する。   
入力欄の"*"はワイルドカードを示す。  
これでCHECKをクリックしてからOKをクリックすると、登録情報に問題がなければドメイン情報の登録は完了する。  

#### IPアドレスの通知  
手元のUbuntuサーバから以下のコマンドを実行してルータのIPアドレスをDDNSサーバに知らせる。MasterIDとPasswordは先ほど受け取ったログイン情報に使ったものを当てはめる。  
```bash
$ wget -o - 'https://MasterID:Password@www.mydns.jp/login.html'
```
IPアドレスを定期的に通知するデーモン(常駐プログラム)を設定する。  
```bash
$ crontab -e
no crontab for username - using an empty one

Select an editor.  To change later, run 'select-editor'.
  1. /bin/nano        <---- easiest
  2. /usr/bin/vim.basic
  3. /usr/bin/vim.tiny
  4. /bin/ed
```

好きなエディタを選択しコメントの最終行に以下のコードを追加する。  
```bash
*/5 * * * * wget -o - 'https://MasterID:Password@www.mydns.jp/login.html'
```  
とりあえずこれで5分ごとにこのコマンドを実行してくれる。  
最後に、ルータとサーバの再起動をする。  
```
$sudo reboot
```
### ログイン  
sshを使ったログインは以下のコマンドで実行可能  
```
$ ssh -p ポート番号 ユーザ名@設定したドメイン名  
```
このコマンドだと**同じネットワーク内で行うとログインできない**ので注意が必要。  

## まとめ
これで他のネットワーク環境からでも自宅のパソコンにアクセスできるようになった。  
手間取ったがなんとかなったので安心。  
固定IP設定が、結局ルータ側からの設定になったが本来はサーバの方で設定をする予定だった。うまくできなかったので方針転換をした。  
## 参考  
- https://futurismo.biz/archives/6862/#%E3%83%AD%E3%83%BC%E3%82%AB%E3%83%ABip%E3%82%A2%E3%83%89%E3%83%AC%E3%82%B9%E3%82%92%E5%9B%BA%E5%AE%9A%E3%81%99%E3%82%8B--ip-  
- [Ubuntu サーバー徹底構築](https://www.amazon.co.jp/gp/product/B07NRPTK6R/ref=ppx_yo_dt_b_d_asin_title_o04?ie=UTF8&psc=1)  
