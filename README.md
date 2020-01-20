### 簡単なFCN作成(とラーニングカーブ)
[こちら](https://arxiv.org/abs/1605.06211)に記述されている
FCNを単純構造で作ったものです.

参考にさせていただいたもの:[MATHGRAM](https://www.mathgram.xyz/entry/keras/fcn)

## Usage

以下のコマンドにより,FCNモデルの訓練と,ラーニングカーブ・学習過程のプロットを行います.

```python:
$ python3 main.py <netCDF file>
```

使用するnetCDFファイルの中身は,**imgX**変数に画像,**imgY**変数に教師用画像(2値)のデータが入っています.
