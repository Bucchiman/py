<!-- FileName: readme
 Author: 8ucchiman
 CreatedDate: 2023-02-28 13:15:12 +0900
 LastModified: 2023-03-24 13:23:24 +0900
 Reference: 8ucchiman.jp
-->


# g-h filter
g: measurement(weight in our example)
h: 


# 用語のおさらい
- 系(system): 推定対象のオブジェクト(測定する対象,人や牛)
- 系の状態(system state): その系に関連づく値のこと(おもりを載せればその値になる)
- 観測値(measurement): 系を観測して得られる値
- 状態推定値(state estimate): フィルタを使って推定した状態の値

  e.g. 体重計に乗って得られた重さは観測値である
       それはセンサーに誤差があるため、状態推定値はまた異なる値となる。
- プロセスモデル

  e.g. 今日の私の体重は昨日の体重に昨日のゲインを足すとえられる

- 系誤差(system error), プロセス誤差(process error)
- 系伝播(system propagation)

  プロセスモデルを使って新しい状態推定値を作る処理


# algorithm

```
    初期条件 (x0)       観測値 (zk)
        |                  |
        |                  |
        v                  v
     +---------+     +---------+
     | 予測    | --> | 更新    |
     | ステップ| <-- | ステップ|
     +---------+     +---------+
        |
        |
        v
        状態推定値(xk)
```
 - Initialization
   1. Initialize the state of the filter
   2. Initialize our belief in the state

 - Predict
   1. Use system behavor to predict state at the next time step
   2. Adjust belief to account for the uncertainty in prediction

 - Update
   1. Get measurement and associated belief about its accuracy
   2. Compute residual between estimated state and measurement
   3. New estimate is somewhere on the residual line



# あらすじ

## 増える体重から明日の体重を予測せよ

### 解決案01
観測値(measurement z)と予測値(prediction x)から推定値()を求める

- 予測値は何らかの仮定をもとに決める

e.g. 以前の推定値から1日で1ポンド増える(ゲイン)と仮定する。

```
    prediction = estimated + gain_rate * time_step
    estimate = prediction + 4/10(measurement - prediction)    
```
これではgain_rateが仮定のもとで定数となり、うまく推定できない
gainを観測値と過去の予測値から得られないか。

### 解決案02
```
    new gain = old gain + 1/3*(measurement - predicted weight) / 1day
```
