# Q. 71 - 80

## Q.71. マスキング

*imori.jpg*に対してHSVを用いて青色の箇所のみが黒くなるようにマスキングせよ。

このように白黒のバイナリ画像を用いて黒部分に対応する元画像の画素を黒に変更する操作をマスキングという。

青色箇所の抽出はHSVで180<=H<=260となる位置が1となるような二値画像を作成し、それの0と1を反転したものと元画像との積をとればよい。

これによりある程度のイモリの部分の抽出ができる。

|入力 (imori.jpg) |マスク(answer_70.png)|出力(answer_71.jpg)|
|:---:|:---:|
|![](imori.jpg)|![](answer_70.png)|![](answer_71.jpg)|

答え >> answer_71.py

## Q.71. マスキング(カラートラッキング＋モルフォロジー)

Q.71ではマスクが雑になってしまっていたので、イモリの目の部分が削除されていたり、背景がところどころ残ってしまった。

よってマスク画像にN=5のクロージング処理(Q.50)とオープニング処理(Q.49)を施してマスク画像を正確にして、マスキングを行え。

|入力 (imori.jpg) |マスク(answer_72_mask.png)|出力(answer_72.jpg)|
|:---:|:---:|
|![](imori.jpg)|![](answer_72_mask.png)|![](answer_72.jpg)|

答え >> answer_72.py