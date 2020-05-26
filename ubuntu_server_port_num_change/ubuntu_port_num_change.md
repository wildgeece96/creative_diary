# Ubuntu Serverでsshのポート番号を変更する  
すぐ忘れそうなのでメモ。  

```
$ sudo vi /etc/ssh/sshd_config  
```
で、
```
Port 10022
```
のように設定し、再起動。  
```
$ sudo systemctl restart sshd
```
