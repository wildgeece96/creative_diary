今日参加した全脳アーキテクチャ勉強会に参加したときのメモをここに残す。  

汎用型AIを作ろうとしている団体WBAIが定期開催している勉強会のようで、今回で第３０回を迎える模様。   

## BCIについての発表(柳沢先生)  

今日の話題は以下の２つ。  

- 侵襲的BCIの医療応用について
- BCIを介した脳とAIのインタラクションについて  

- Brain Machine Interfaceについて  
- 脳波から機械を動かす

NHKでまとめられたビデオなどを見せてもらって、映像での仕組みを簡単に見せてくれた。  
電極を脳の中に埋め込むのはすごいなと思ったが、ノイズの量を考えると頭の外側につけるよりかはかなり現実的な印象を受けた。  
ものを握ろうという信号を応用してパソコン操作をできるように工夫してる。  

### 医療応用について  

- 64チャネルの電極を頭蓋に埋め込んで脊髄損傷の患者の移動を手助けすることができた  

現在、ALSの患者に向けたBCIを用いた文字入力の速度は31character per minutesらしい。確かにスムーズなコミュニケーションを送るには不十分そうな速度。。   

#### 脳波から直接音声生成  

それを受けて、音声を脳波から生成する研究がある。脳波から口の動きを予測し、予め学習された口の動きと音声との対応関係から音声を生成するという研究もある。  
え、すごい。。。
何となく聞こえる。  

#### 脳波からText生成  

これにも深層学習を使用。  
RNNをベースに翻訳をしている。(seq2seqモデルを使っている様子)  

ところどころ出てくる $high-\gamma$ はどんな種類の脳波なのだろうか。。。  

精度もタイプライターの方の精度と遜色ないものになった。  

#### 脳情報の解読  

これらのように非常に精度が高まってきている。  
読み取りとして機械学習の技術が頻繁におうようされているみたい。  

#### 脳へ情報を入力する

脳から読み取るだけでなく、脳へ情報を入力する技術も研究されている。  

蓋膜下電極をつかっている。  

医療応用の例  
- てんかんの発作に対して、頭の中に留置した電極からてんかん発作を検知して、脳を電気刺激して発作をとめるデバイス  
- パーキンソン病。  

#### 脳波の解読をより精密にする

気分障害(うつ)に対して電気刺激で気分を制御する？  
　→　すごそう。ディストピア感のある技術ですね。  

### BCIの行きつく先  

BCIを通じて機能の補充をすることにスポットが当てられてきたが、これを拡張していけば脳機能の拡張であったり汎用型AIへつなげることができるのではないか。  

#### Neurallink  

イーロンマスクさん
多チャンネルで計測ができ、脳への入力も可能なデバイス  
モチベーションの１つとして脳機能の拡張がある  

### 課題

- BCIを介して脳とAIがインタラクションすると何ができうるのか。  

掲げられている課題を解決するために、JST CRESTで研究チーム(?)が立ち上げられている。  
神経科学、脳外科の２チームに別れている。  

頭蓋内脳波のビッグデータを作ろうとしている。これはわくわくする。  

### 視覚処理について  

視覚処理においてはDNNは最も良いモデル。非常に脳の活動に似ているということが知られている。  

どれぐらいCNNの活動状況が脳に似ているのかを対応付けた値: Brain-Score  
ImageNetのパフォーマンスとBrain-Scoreには正の相関がある・・！  

(Neural Encoding)  
猫の入力で、V4と呼ばれる視覚処理部分における活動状態をシミュレートするという実験もある。  

(Neural Decoding)  
実際に視覚野で見たものを、脳波から予測することが出来た。  
思い浮かべるだけでも可能。  
GANを用いて、視覚情報を再構成するという研究も行われている。  
→　ここまで来ると、夢の可視化もだんだん現実味を帯びてくる気がする。  

### 幻肢痛に対するアプローチ  

失われた四肢から痛みがやってくる(幻の痛み)  
幻肢を動かせるようなBCIをセットすると痛みが軽減するという実験結果が。 
デコーダの種類によって痛みの軽減度合いが変わってくる。(健常肢用のデコーダが最も痛みを軽減した。幻肢用のデコーダが痛みが最も残った)  

これについては、非侵襲型でもよいらしい。  

## ヒューマンエージェントインタラクション：AIとHCIの葛藤  (今井先生)  

ヒューマンインタフェイスは歴史上でAIが持つ機能、知能を捨てていくことで発展してきた。現状ではAIが持つ知能がインターフェイスに耐えうるものになってきている。これらを踏まえると、これまでのインターフェイスが実現してきたインタラクションを保持しつつ、知能部分を作成するというのは難しくないという内容の発表内容であった。  

インターフェイスが知能をこれまで捨ててきたという見方は新しかった。  

## ディスカッション  

- 知識をグラフ形式で表現できたらより複雑な知識グラフが脳内に埋め込まれている可能性がある  
- 全脳のエミュレーションはハエやネズミレベルであればできていると言われている。計算レベルではここ2,30年で人間レベルになるのではないか。  
