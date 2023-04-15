<!-- FileName: readme
 Author: 8ucchiman
 CreatedDate: 2023-02-13 14:09:40 +0900
 LastModified: 2023-02-13 14:26:03 +0900
 Reference: https://qiita.com/HiromuMasuda0228/items/a7a861796dfeac604f47
-->


# DataFrameとSeries
DataFrameは二次元のデータ構造, Seriesは一次元のデータ構造である。
```python
    $ train_df[['amount']]
    > DataFrame
    $ train_df['amount']
    > Series
```

DataFrameからデータを抽出し、一次元の結果になった時、Seriesとして結果を出力
一方で、Seriesからデータを抽出し、結果を出力すると、valueとして結果を出力

```python
    $ train_df[['amount']].nunique()
    > Series
    $ train_df['amount'].nunique()
    > int
```
