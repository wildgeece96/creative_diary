javascriptで複数の要素を選択して操作する方法

## getElementsByClassNameは配列を返す  
html内のjavascriptの要素指定で、
`getElementsByClassName("<クラスネーム>")`といったものがあるが、これによってそのクラスに該当する要素が配列になって渡されるらしい。  

なので、  
`document.getElementsByClassName("<クラスネーム>;").innerHTML = "";`のように一括して複数要素変えることはできなくて、配列の中身を1つずつ変えていく処理にするしかなさそう。  
### forループに入れて処理をする  
```
var elements = document.getElementsByClassName("<クラスネーム>");
for(var i=0;i<elements.length;i++){
  elements[i].innerHTML = "";
  elements[i].style.fontSize = "large";
}
```
上記のようなforループでスタイルや文章の中身を変える必要がありそう。  
以下にボタンを押したら文字がでかくなるhtmlファイルを書いた。  

```
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
    <p class="text">ここのテキストが</p>
    <p class="text">まとめて</p>
    <p class="text">大きくなるよ</p>
    <button id="makeBig">文字を大きくする</button>

    <script type="text/javascript">
      document.getElementById("makeBig").onclick = function(){
        var texts = document.getElementsByClassName("text");
        for(var i=0;i<texts.length;i++){
          texts[i].style.fontSize = "50px";
        }
      };
    </script>
  </body>
</html>

```

これをブラウザで開くと以下のようになる。   
<img src="https://leck-tech.com/wp-content/uploads/2018/08/before.png" alt="" width="539" height="493" class="alignleft size-full wp-image-197" />  
ここの下にあるボタンを押すとめちゃ文字が大きくなる。    
<img src="https://leck-tech.com/wp-content/uploads/2018/08/after.png" alt="" width="537" height="499" class="alignleft size-full wp-image-196" />

複数要素に対して操作ができるのはコードの簡略化にも使えるので覚えておくと便利。  
