<!-- FileName: readme
 Author: 8ucchiman
 CreatedDate: 2023-02-06 16:26:17 +0900
 LastModified: 2023-02-09 16:29:30 +0900
 Reference: https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9
-->


# 前提知識
![image](https://matplotlib.org/1.5.1/_images/fig_map.png)

![term](https://matplotlib.org/stable/_images/anatomy.png)

```python
    fig, ax = plt.subplots()   # Figureオブジェクトとそれに属する1つのAxesオブジェクトを同時に作成
```[^1]

[^1]別の書き方
```python
    fig = plt.figure()  # Figureオブジェクト作成
    ax = fig.add_subplot(1, 1, 1)  # Axesオブジェクト作成
```
