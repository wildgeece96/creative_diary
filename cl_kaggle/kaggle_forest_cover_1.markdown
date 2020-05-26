
今回は現在開催中のKaggleのコンペティションの内容を備忘録がてら記録する。  


Kaggle自体は前回のAvitoで少し他の人のコードを真似したレベルでまだまだ初心者なので、今回は賞金のついていない、簡単なものをやることに。  

## 今回扱うコンペティション  
それで今回選んだのはこちら。  

[Forest Cover Type](https://www.kaggle.com/c/forest-cover-type-kernels-only)  

どうやらこのデータセットはコロラド州立大学から提供されたもののよう。  
機械学習の練習台枠としてKaggleないで開催されているようだ。  

### 何を求めるのか  
今回は、与えられたデータから森林にある木の種類判定を行うのが目的らしい。  

この7つの種類にわけるようだ。  

>
>1 - Spruce/Fir  
>2 - Lodgepole Pine  
>3 - Ponderosa Pine  
>4 - Cottonwood/Willow  
>5 - Aspen  
>6 - Douglas-fir  
>7 - Krummholz  
>

上から  
1 トウヒ  
![spruce.jpg](../img/kaggle/forest_cover_type/spruce.jpg)  
##### https://www.thespruce.com/twelve-species-of-fir-trees-3269663 より  
2 コントルタマツ  
![lodgepole_pine.jpg](../img/kaggle/forest_cover_type/lodgepole_pine.jpg)  
##### http://treetime.ca/productsList.php?pcid=88&tagid=2 より  
3 ポンデローザマツ  
![ponderose_pine.jpg](../img/kaggle/forest_cover_type/ponderosa_pine.jpg)   
###### https://www.tnnursery.net/ponderosa-pine-trees-for-sale/ より    
4 ヒロハハコヤナギ(ポプラの一種?)
![cottonwood.jpg](../img/kaggle/forest_cover_type/cottonwood.jpg)   
##### https://www.tnnursery.net/cottonwood-tree-for-sale/ より  
5 ヤマナラシ(ポプラの一種?)  
![aspen.jpg](../img/kaggle/forest_cover_type/aspen.jpg)  
6 ベイマツ  
![douglas.jpg](../img/kaggle/forest_cover_type/douglas.jpg)
##### http://www.forevergreenchristmastree.com/new-products/copy-of-douglas-fir より  
7 スイスコウザンマツ  
![krummholza.jpg](../img/kaggle/forest_cover_type/krummholz.jpg)  
##### https://en.wikipedia.org/wiki/Krummholz より  
なんかかっこいいな。  

### 使うデータ  
ルーズベルト国立森林公園(和名適当です)において30メートル四方における観測データを元に先ほどの木の種類を推定します。  

トレーニングデータは**15120**個で、テストデータは**565892**個らしい。  
(テストデータ多いな)  

与えられるデータは以下の表の通り。  

|データ名|概要|
|:--|:----:|
|Elevation|標高(m)|
|Aspect|向いている方位|
|Slope|傾斜|
|Horizontal_Distance_To_Hydrology|水場までの二次元上(地図上)の距離|
|Vertical_Distance_To_Hydrology|水源までの標高差|
|Horizontal_Distance_To_Roadways|最も近い道路までの二次元上の距離|
|Hillshade_9am(0から255までのインデックス)|陰影起伏のインデックス(午前9時)|
|Hillshade_Noon(0から255までのインデックス)|陰影起伏のインデックス(正午)|
(Hillshade_3pm(0から255までのインデックス)|陰影起伏のインデックス(午後3時)|
|Horizontal_Distance_To_Fire_Points|山火事が起きたポイントまでの水平距離|
|Wilderness_Area(4つの2値のカラム, 0=なし,1=あり)|荒野の区分|
|Soil_Type(40の2値のカラム,0=なし,1=あり)|土の性質|
|Cover_Type(1から7までの整数)|今回求めるもの。観測されたエリアに生えている木の種類|

雑だが、訳すとこんな感じになった。  

Soil_TypeとWilderness_Areaにはそれぞれラベルが40と5つあり、それぞれのラベルに当てはまれば`1`を、当てはまらなければ`0`が入っているらしい。  
これは扱いやすそう。  
### データの中身をのぞく   
[公式サイト](https://www.kaggle.com/c/forest-cover-type-kernels-only/data)からtrain.csvをダウンロード。これがトレーニングデータとなる。  

とりあえず今回は中身を少しだけ確認するだけにとどめる。  
学習などは次回以降にやるつもり。  

PythonのPandasを使って中身を見る。  
```python


In [1]: import pandas as pd

In [2]: train = pd.read_csv("train.csv")

In [3]: train.head()
Out[3]:
   Id  Elevation  Aspect     ...      Soil_Type39  Soil_Type40  Cover_Type
0   1       2596      51     ...                0            0           5
1   2       2590      56     ...                0            0           5
2   3       2804     139     ...                0            0           2
3   4       2785     155     ...                0            0           2
4   5       2595      45     ...                0            0           5

[5 rows x 56 columns]

In [4]: train.iloc[0,:]
Out[4]:
Id                                       1
Elevation                             2596
Aspect                                  51
Slope                                    3
Horizontal_Distance_To_Hydrology       258
Vertical_Distance_To_Hydrology           0
Horizontal_Distance_To_Roadways        510
Hillshade_9am                          221
Hillshade_Noon                         232
Hillshade_3pm                          148
Horizontal_Distance_To_Fire_Points    6279
Wilderness_Area1                         1
Wilderness_Area2                         0
Wilderness_Area3                         0
Wilderness_Area4                         0
Soil_Type1                               0
Soil_Type2                               0
Soil_Type3                               0
Soil_Type4                               0
Soil_Type5                               0
Soil_Type6                               0
Soil_Type7                               0
Soil_Type8                               0
Soil_Type9                               0
Soil_Type10                              0
Soil_Type11                              0
Soil_Type12                              0
Soil_Type13                              0
Soil_Type14                              0
Soil_Type15                              0
Soil_Type16                              0
Soil_Type17                              0
Soil_Type18                              0
Soil_Type19                              0
Soil_Type20                              0
Soil_Type21                              0
Soil_Type22                              0
Soil_Type23                              0
Soil_Type24                              0
Soil_Type25                              0
Soil_Type26                              0
Soil_Type27                              0
Soil_Type28                              0
Soil_Type29                              1
Soil_Type30                              0
Soil_Type31                              0
Soil_Type32                              0
Soil_Type33                              0
Soil_Type34                              0
Soil_Type35                              0
Soil_Type36                              0
Soil_Type37                              0
Soil_Type38                              0
Soil_Type39                              0
Soil_Type40                              0
Cover_Type                               5
Name: 0, dtype: int64


```

全て数値データとなっているので扱いやすそうだ。  

## 作戦  
RIDGEとかlightGBMくらいしか思いつかない(前回のAvito Demand Predictionで主流の手法だったため)くらいの知識のなさなのでDiscussionをのぞいてどのような手法があるか調べた方がよさそう。  

まずは自分なりに少し実装してから色々調べるようにしたい。  


## まとめ  
今回はKaggleの**Forest Cover Type**というコンペティションの概要を説明した。  

次回からは色々とデータをいじっていきたい。  
