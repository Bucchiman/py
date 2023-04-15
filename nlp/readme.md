<!-- FileName: readme
 Author: 8ucchiman
 CreatedDate: 2023-02-18 11:37:23 +0900
 LastModified: 2023-02-19 16:32:49 +0900
 Reference: 8ucchiman.jp
-->



# 単語の意味を理解する手法
 - シソーラス(thesaurus)
 - カウントベース
 - 推論ベース(word2vec)


## thesaurus
広辞苑のような類語辞書としてマシンに理解させる

```
  car = auto automobile machine motocar
```

- WordNet

プリンストン大学によって開発された伝統あるシソーラス

### 問題点
- 時代の変化に対応しにくい
- 人手の面でコストが高い
- 単語のニュアンスを表現できない

## countbase
コーパス(corpus)の利用

corpusとは大量のテキストデータのこと
[code](./make_corpus.py)

単語の関係性を表す手法として、従来からシンプルなアイディアである
「単語の意味は、周囲の単語によって形成される」
という分布仮説によって研究がされてきた。

### 共起行列
分布仮説に基づく単語をベクトルで表現する。
シンプルな考え方は、周囲の単語をカウントすることです。
[code](./make_co_occurence_matrix.py)

### コサイン類似度
共起行列から単語のベクトルを用いて単語間の類似度を調べ、類似性を見ます。
```math
 similarity(x, y) = \frac{x\dot y}{\lVert x}
```

## improve countbase
周囲の単語をカウントするアイディアでは、"the"と"car"が強い関係性があるということになる。
一方で、"drive"と"car"は明らかに関連性があるが、単純なカウントでは、"the"と"car"の方が関連性が強いことになってしまう。
改善方法として、相互情報量(Pointwise Mutual Information)がある。
### PMI, PPMI
定義式は次の通り。
```math
  PMI(x, y) = \log2\frac{P(x, y)}{P(x)P(y)}
```


### 特異値分解
次元削減手法の一つ

特異値分解(Singular Value Decomposition: SVD)を使い、任意の行列を３つの行列の積へと分解する。
```math
  X = USV^T
```
UとVは直交行列、その列ベクトルは互いに直交する。
Sは対角行列、対角成分以外全て0の行列
Uはなんらかの空間の軸を形成する。
Sは対角行列で、対角成分には「特異値」というものが大きい順に並んでいる。
これを、対応する軸の重要度とみなせる。
Sの対角成分から重要な軸を選択し、Uから余分な列ベクトルを削ることができる。


