# Homebrewを再インストールした後の復旧方法(Mac OS Mojave, zsh)  

`brew install`のオプションがうまく使えない？と思ったので試しに再インストールしてみたら色々壊れてしまい、復旧作業をする必要があった。  
今回はその時のログを残す。  
anacondaをなぜか使わない主義になってしまったので、pyenv主導の環境に設定する。  
環境は  
```
OS : macOS Mojave (ver 10.14)
```
## Homebrewの再インストール  
まずはHomebrewの再インストール(これが諸悪の根源)  
やるときは気をつけてほしい。  
warningでHomebrewで今までインストールされたパッケージが全て失われると書いてあったので警告はされていた。  

#### homebrewのアンインストール
```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
```
#### homebrewのインストール  
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
## zshのインストール  
brewコマンドを使ってzshを入れる  
```
brew install zsh
/usr/local/bin/zsh --version
chsh -s /usr/local/bin/zsh
$SHELL --version
```
最後ので  
```
zsh 5.7.1 (x86_64-apple-darwin18.2.0)
```
のような結果が返されればOK。  
最後に、PATHを通しておく。  

## pyenvのインストール  
pythonを入れるためにbrewコマンドを使って入れ直す  
バージョンを適宜入れ替えられるように、pythonを直接インストールするのではなく、pyenvを先にインストールする。  
```
brew install pyenv  
```
PATHを追加するために、`.zshrc`をいじる。  
(`.zshrc`はホームディレクトリ直下に配置する)  
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
これを読み込ませる  
```
source .zshrc  
```
### pythonのインストール  
```
pyenv install --list
```
でインストールできるpythonのバージョン一覧を確認する。  

とりあえず、`3.7.3`が最新っぽいのでこれをインストール。  
```
pyenv install 3.7.3
```
#### エラー  
エラーが返ってきた。  
```
Inspect or clean up the working tree at /var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649
Results logged to /var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649.log

Last 10 log lines:
  File "/private/var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649/Python-3.7.3/Lib/ensurepip/__main__.py", line 5, in <module>
    sys.exit(ensurepip._main())
  File "/private/var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649/Python-3.7.3/Lib/ensurepip/__init__.py", line 204, in _main
    default_pip=args.default_pip,
  File "/private/var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649/Python-3.7.3/Lib/ensurepip/__init__.py", line 117, in _bootstrap
    return _run_pip(args + [p[0] for p in _PROJECTS], additional_paths)
  File "/private/var/folders/mj/tq3l77y12694xfyylxswxjkm0000gn/T/python-build.20190521145610.67649/Python-3.7.3/Lib/ensurepip/__init__.py", line 27, in _run_pip
    import pip._internal
zipimport.ZipImportError: can't decompress data; zlib not available
make: *** [install] Error 1
```
#### 上記のエラーの解決策
[[MacOS Mojave]pyenvでpythonのインストールがzlibエラーで失敗した時の対応](https://qiita.com/zreactor/items/c3fd04417e0d61af0afe)  
を参考にした。  
原因は  
>xcode-selectの最新バージョン(2354)にMojave用のmacOS SDK headerがデフォルトで入っていないのが原因のようです。  

とのことなので、  
以下のコードを実行すると解決するとのことだったので実行してみた。  
```
sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /
```
もう一回インストールしてみる。  
```
pyenv install 3.7.3
pyenv global 3.7.3
```

パスが通っていれば、  
pip3が使えるようになっているはず。  

```
which pip3  
# /Users/<username>/.pyenv/shims/pip3
```
