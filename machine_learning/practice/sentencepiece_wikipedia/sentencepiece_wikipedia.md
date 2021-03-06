# 日本語Wikipediaデータを使ったsentencepieceを学習させる  

環境は  
```
Ubuntu 18.04 (LTS)

```
## Wikipediaデータの準備  
以下のサイトを参考に進めていたが、若干勝手が違っていたのでここでもメモっておく。  
[Wikipediaからコーパスを作る](http://kzkohashi.hatenablog.com/entry/2018/07/22/212913)
### ダウンロード  
とりあえず以下のコードでダウンロードする。  

```shell
$ curl https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 -o jawiki-latest-pages-articles.xml.bz2
```

### txtファイルへの変換  
[Wikiextractor](https://github.com/attardi/wikiextractor)を使用する。  

```shell
$ git clone https://github.com/attardi/wikiextractor
```
wikiextractorフォルダ以下の`Wikiextractor.py`を実行すれば良い。(なぜかREADMEのsetup.pyが見当たらないのでinstallはやらない)  

```shell
$ python Wikiextractor.py -o /path/to/output/directory --processes 5 /path/to/wiki/jawiki-latest-pages-articles.xml.bz2
```
`--processes 5`は適宜選ぶ。（数字が大きいほど処理は早そう)  
`-o`オプションで変換したファイルの保存を行う。  
最後に変換するファイルの場所を指定する。  


これをすると、以下のようなフォルダが指定したディレクトリの所に生成される。  

```
AA  AB  AC  AD  AE  AF  AG  AH  AI  AJ  AK  AL  AM  AN  AO  AP  AQ  AR  AS  AT  AU  AV  AW  AX  AY  AZ  BA  BB
```



### ファイルを1つのテキストファイルにまとめる   
ではこれらのファイルをひとまとめにする。（する必要もないかもだが）  
データが生成されたディレクトリに移動して、以下のコマンドを実行。  

```
$ find */ | grep wiki | awk '{system("cat "$0" >> wiki.txt")}'
```

### <doc>タグの削除  
中身をみてみる。  
```
$ head wiki.txt
```
```
<doc id="19064" url="https://ja.wikipedia.org/wiki?curid=19064" title="マリー・アントワネット">
マリー・アントワネット

マリー＝アントワネット＝ジョゼフ＝ジャンヌ・ド・アブスブール＝ロレーヌ・ドートリシュ（, 1755年11月2日 - 1793年10月16日）は、フランス国王ルイ16世の王妃。フランスの資本主義革命「フランス革命」で処刑された。マリア・テレジアの娘であり、「美貌、純情な反面、軽率、わがまま」だったとされており、乱費や民衆蔑視によって国民から反発されていた。ベルサイユの宮廷生活を享楽し、その浪費などから「赤字夫人」「オーストリア女」と呼ばれた。アントワネットはさまざまな改革に常に反対し、また青年貴族たち（特にH.フェルセン）との情愛に溺れたことで「軽率、浪費家」だったと現在では評価されている。1785年の王妃をめぐる、無実の詐欺事件「首飾り事件」も、結果的に国民の反感へとつながった。

1789年のフランス革命に反対し、宮廷の反革命勢力を形成したアントワネットは、立憲君主制派（ミラボーやラファイエットなど）へ接近することさえも拒んだ。君主制維持を目的として武力干渉を諸外国に要請し、特にウィーン宮廷との秘密交渉を進め、外国軍隊のフランス侵入を期待した。しかしヴァレンヌ逃亡に失敗、反革命の中心人物として処刑された。フランス革命を代表とする資本主義革命（ブルジョア革命）は、身分制・領主制といった封建的な残留物を一掃し、資本主義の発展および資本主義憲法（典型例としてフランス憲法）の確立を成し遂げた。

1755年11月2日、神聖ローマ皇帝フランツ1世とオーストリア女大公マリア・テレジアの十一女としてウィーンで誕生した。ドイツ語名は、マリア・アントーニア・ヨーゼファ・ヨハンナ・フォン・ハプスブルク＝ロートリンゲン。イタリア語やダンス、作曲家グルックのもとで身につけたハープやクラヴサンなどの演奏を得意とした。3歳年上のマリア・カロリーナが嫁ぐまでは同じ部屋で養育され、姉妹は非常に仲がよかった。オーストリア宮廷は非常に家庭的で、幼いころから家族揃って狩りに出かけたり、家族でバレエやオペラを観覧したりした。また幼いころからバレエやオペラを皇女らが演じている。

当時のオーストリアは、プロイセンの脅威から伝統的な外交関係を転換してフランスとの同盟関係を深めようとしており（外交革命）、その一環として母マリア・テレジアは、自分の娘とフランス国王ルイ15世の孫、ルイ・オーギュスト（のちのルイ16世）との政略結婚を画策した。当初はマリア・カロリーナがその候補であったが、ナポリ王と婚約していたすぐ上の姉マリア・ヨーゼファが1767年、結婚直前に急死したため、翌1768年に急遽マリア・カロリーナがナポリのフェルディナンド4世へ嫁ぐことになった。そのため、アントーニアがフランスとの政略結婚候補に繰り上がった。
```
文章の先頭に <doc> タグがあることがわかる。  
これを取り除く  
雑ではあるが`<doc`と`</doc`のある単語を削除する。  
```
$vim wiki.txt
```
多少重いがそこは強引に。  
ここで以下の2つのコマンドを実行する。  

```
:g/<doc/d
:g/<\/doc/d
```  

## sentencepieceのモデルを学習させる  
時間がかかりそうだったのでC++でインストールしたらハマったのでPythonのを使って無理やり学習させる。  

```
$pip install sentencepiece
```

以下のスクリプトを作成して学習をする。  
(モデルの保存用にsen_modelディレクトリを作成した)  
全部の文章を使うことはできなかったので、入力されたうちの約半分(1000万センテンス)ほどを学習に使うことにした。  

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train("--input=wiki.txt --model_prefix=sen_model/wiki_model_32_all --vocab_size=32000 --character_coverage=1.0 --input_sentence_size=10000000\
                   --seed_sentencepiece_size=10000000 --shuffle_input_sentence=true")
```
`vocab_size`の値を変えた結果も保存しておきたい。  
全部の

### 動作確認  
以下のコードで動作確認。  
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('sen_model/wiki_model_32_all.model')
text="僕の周りには分からないことだらけだけれど、1つずつ理解していくような努力をしていきたいと思う。"
print(sp.EncodeAsIds(text))
print(sp.EncodeAsPieces(text))
```

結果は以下のような感じ。  
```
[6, 5383, 5, 7186, 43, 18728, 308, 330, 172, 1039, 791, 612, 1194, 3, 21, 176, 2915, 3166, 3309, 1083, 7334, 11, 28, 11559, 866, 11616, 4]
['▁', '僕', 'の', '周り', 'には', '分からない', 'こと', 'だ', 'ら', 'け', 'だけ', 'れ', 'ど', '、', '1', 'つ', 'ずつ', '理解', 'していく', 'ような', '努力', 'を', 'し', 'ていき', 'たい', 'と思う', '。']
```
直感的な不自然さはなさそう。  
このように、それぞれに対応するIDを出力させることも可能になる。  

これを自然言語処理タスクのインプットに使って何かしら学習させてみたい。    
