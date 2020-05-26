javascriptの変数の型一覧と使い方まとめ  

多分色々なサイトでまとめられているとは思うが、自分なりにも一通りまとめてみる。  

javascript自体は動的型付けの言語なので、予め型宣言をしなくても変数を宣言することが可能。  

## 変数の作り方  
変数の作り方自体は簡単で、変数としたいものの前に`var`をつけてそれに何かしらの値を代入すれば完了。  
[javascript]
var x = 2;
var text = "sample text";
var arr = ["A", "b", "cc"];
[/javascript]
こうすると勝手に型を推定してくれる。  

変数の型を上書きすることも可能。  
[javascript]
var y = 3; // ここではNumber
var y = "text"; // ここではString
[/javascript]

## 型一覧と最初の宣言の仕方  
とりあえずその型を入れるための入れ子を予め作っておきたいときがある。  
例えば文字列を入れていきたいときは
[javascript]
var str_1 = "";
[/javascript]
とすると文字列型として初期化することができる。他の型についても同じような記法があるので以下にまとめていく。  
#### primitiveデータ
- String型
- Number型
- Boolean型
- Undefined  
#### complexデータ
- 関数(function)
- object型

#### その他
- Null   
- Object型  
- Array型

それでは1つずつ簡単に見て行く。  

### String型  
いわゆる文字列を入れてくれる型
[javascript]
var text = new String; // 中身が入っていないstring型
var text = "This is a sample text.";
[/javascript]

### Number型  
数字を入れてくれる  
64ビット形式で処理しているらしい。  

[javascript]
var num = new Number; // Number型の宣言
var num_1 = 123;
var x = 1e-5; // 0.00001
[/javascript]

### Boolean型  
True or False の2つの値だけ持つ。  
[javascript]
var value = new Boolean;
var value2 = true;
var value3 = false;
[/javascript]

### Undefined
何も型宣言がされていない状態。  
型も値も決まっていない。
[javascript]
var x; //これだけにするとundefinedになる
[/javascript]

### Null型  
型だけが決まっている状態で値が何も入っていない状態のもの。  
持っている値は`null`のみ。  
[javascript]
var num = 2;
num = null; // このとき値はnullになるが型はNumber
[/javascript]

### Array型
大きな枠ではObject型に分類されるが、特によく使われる。  
複数の同じ型のものを入れることができる。  

[javascript]
var arr = new Array;
var arr2 = ["a", "bb", "D"];
var arr3 = [2, 3, 1, 5];
[/javascript]
配列のアクセスは左から何番目(最初は0から)にあるのかでアクセスできる。  
[javascript]
alert(arr3[0]); // 2が表示される
alert(arr2[1]); // bbが表示される
[/javascript]

### Object型  
Pythonで言うとdictと同じ扱い。  
ある特定のキーに値を対応させるもの。  
配列の場合は番号だったが、それがキーになる。  
[javascript]
var ob = new Object;
var ob2 = {firstName:"John", lastName:"Doe", age:50, eyeColor:"blue"};
[/javascript]
それぞれのキーにアクセスするには`.key`でアクセス。  
[javascript]
alert(ob2.firstName); // Johnが表示される。
[/javascript]

### function型  
関数。特定の処理を記録してくれる。  
[javascript]
function func(a,b){
  return a+b;
}
[/javascript]

### NullとUndefinedとの違い  
NullとUndefinedは似てるように見えるが、明確な違いがあり、それは **型が決まってるかどうか** である。  
Nullについては **型が決まっている**　上で中身がないのであるが、  
Undefinedは **型も中身も決まっていない** 状態になっている。  

## まとめ

PythonとC言語などとあまり変わらない型ばかりだった。
若干ネーミングが違うのと数値データの精度を変えられるのかどうかがわからなかったが、恐らく計算速度が求められる場面がないことが原因なのかなと勝手に推測。   

## 参考  
- [JavaScript のデータ型とデータ構造 - JavaScript | MDN](https://developer.mozilla.org/ja/docs/Web/JavaScript/Data_structures)
- [JavaScript Data Type - W3School](https://www.w3schools.com/jS/js_datatypes.asp)
