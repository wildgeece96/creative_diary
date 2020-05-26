タイトル　htmlファイルにJavascriptコードを挿入する方法  


今回はhtmlファイルにJavascriptのコードを挿入する方法をまとめてみた。  

## htmlファイルに直接挿入  

### headタグの間にコードを挿入  

以下のように、`<head>`タグの間にJavascriptのコードを挿入することが可能。  

[html]
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
    <script type="text/javascript">
      // Javascriptはここにもかける
    </script>
  </body>
</html>
[/html]

### bodyタグの間にコードを挿入  

同様の形式で`<body>`タグの間に挿入することも可能。  

ここで注意なのはJavascriptのコードは上から読み込んで行く際、 **htmlコードの読み込みを中断して** Javascriptのコードを実行するから、Javascriptのコードは **bodyタグの最後** に挿入するのが無難らしい。  

[html]
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
    <script type="text/javascript">
      // Javascriptはここにもかける
    </script>
  </body>
</html>
[/html]


## 外部ファイルから挿入  

Javascriptファイルを読み込む形式がある。  
これだとソースコードの分割ができるため、Webサイトの管理がしやすくなるのと、いちいち同じソースコードを書かなくても同じJavascriptのファイルを参照すれば同じ処理を再現できるので便利。  

[html]
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
    // ここにJavascriptのファイルを挿入する
    <script type="text/javascript" src="sample.js"></script>
  </body>
</html>
[/html]
