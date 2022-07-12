# ML_projects

## Machine Learning Algorithm Implementation

## 1. BackPropogation
## BackPropogation Algorithm Implementation
![Variable Declaration](/mlimg/backp.jpeg)
## Report
* Basic: *
一開始在寫initialize parameter的部分時，不太懂為什麼要用 HE initialization，後來才發現它 跟準確率有很大的關係，這次作業從一開始就使用到大量的numpy array，也花了一些時間去 熟悉語法的部分。在做activation funtion layer時，也想了一下該怎麼用比較漂亮的寫法寫條 件判斷array中個別的元素，才能避免exponential overflow。
在L model forward中，將(wa+b)傳回linear activation forward，用sigmod讓資料介於0與1之間 來實作binary classifier，要特別注意sigmoid需要用stable function判斷大於0或小於0的情況來 避免exponential overflow。
一開始dimension跟node設比較大，但發現太多node反而cost變高，還有試過各種learning rate，發現0.01下降速度最快;將iteration次數慢慢調成10000，就可以達到很好的效果了。
將basic的部分一步一步時做完後即對整個neutral network的運作原理有更深入的了解。從一 開始初始化參數，接著跟將x跟權重相乘向前傳遞做liner forward。中間有三種activation function layer，分別為sigmoid, softmax, relu三種處理資料的函數，在linear activation forward 向前傳遞資料。中間會計算cost，因為在稍後的backpropogation中，cost會扮演重要的角色， 也是多層neutral network的精神所在。接著，在傳回過程中，我們將cost傳回並求得sigma， 與hidden unit的之間的權重相乘，再一路傳回最底層，將學習到的資訊與learning rate相乘更 新權重，用gradient decendent的方法越來越逼進目標結果。接著嘗試不同dimension，發現其 實不是越高越好，經過3.中的測試，如果太多layer node的話效能會變差，所以設定dimension 為[4,1]，learning rate = 0.01，iteration 數 = 10000，最後達到1.0的accuracy。
* Advanced - Using Softmax
寫softmax的時候發現不能用一般寫法，還要考慮會不會overflow的問題，後來在 backwardpropogation 的softmax卡了有點久，後來才想通為什麼dZ = s-Y，也真正了解到用 cache存值的意義。再寫bonus model的時候dimension寫錯，後來改正input跟output的數量後 才順利訓練好model。
在L model forward中，將(wa+b)傳回linear activation forward，用softmax來實作multi-class classifier，softmax則要記得減掉max值以避免出現很大的數字。
先試[64, 1, 1, 1, 4]，發現accuracy只有25趴後，改成用4dimension，hidden-layer從1往上加， 發現設成3最好，當iteration = 2000，hidden-layer = [3,2]時，cost = 1.09cost; 當iteration = 2000，hidden-layer = [3,4]時，cost = 1.23; 最後，當iteration = 2000，hidden-layer = [3,3]時 ，cost = 1.04，結果最好且有在穩定的下降，所以將iteration一次增大5000來測試幾次的效果 最好。learning rate則是越大下降越快，所以設成0.01。接著發現iteration超過10000之後下降 的速度變慢，但越多圈cost還是有小幅度的下降，20000圈時，accuracy = 0.627，30000圈時 = 0.6292。到50000圈後，下降速度為每5個iteration/0.000001的cost ; 而到90000圈後，下降速 度為每13個iteration/0.000001的cost。最後決定慢慢訓練到100000個iteration，將accuracy提升 至0.6295。
bonus的部分，我將原本的訓練資料拆成x_train與x_val, y_val,與y_train，用來驗證model的 accuracy。

## 2. CNN Cancer Detect
## Project Detect Cancer from Patiant's Lung Image
![Variable Declaration](/mlimg/cnn.png)
## Report
* Basic:
2022/6更新:
原先f1score 沒達到標準應該是因為filter數太少，所以將model filter增加，改成三層分別為 32,256,256，另外增加stride為來控制萃取的大小與速度，並在最後用validation data測試 f1score是否有達標，經過測試至少可以達到0.7。
原文：
使用三層神經網路，filter分別為16,32,64，3x3的卷積核萃取圖片資料，中間摻雜maxpool與 dropout防止過擬合，最後將dense size合為1，因為basic的output dimension是Edema 一個 column。
optimizer用adam，讓learning rate 可以更加平穩，因為他會保留過去的momentum和矯正 偏離，loss用binary_crossentropy，因為這是binary classification問題。
1.原本遇到loss太⾼高，accuracy太低的問題，最後使⽤用Drop out有改善許多。
2.要將input size 做Reshape成(10000,128,128,1)，加上最後⼀一個顏⾊色的參參數，不然size會不 合沒辦法訓練模型。
3.loss要⽤用.binary Crossentropy⽽而不是catagory，因為這是binary classification不是multiple classification問題，是判斷單⼀一label是0還是1⽽而不是分類問題，label們都有可能為0或1。 4.Dense number也是印出來來看了了很多次確認最後有沒有輸出正確的output size。 5.要加上y_train = y_train.reshape((-1,1))才能成功輸入
6.activate function⽤用Sigmoid跟softmax的結果差很多，sigmoid才是適合binary classification。
根據以上注意事項將x_train y_train reshape調整input size後輸入神經網路模型中萃取圖片資 料，訓練後再將test資料輸入模型得到output預測結果並印出csv檔案。
* Advanced - 7 binary classification
使用三層神經網路，filter分別為16,32,64，3x3的卷積核萃取圖片資料，中間摻雜maxpool與 dropout防止過擬合，最後將dense size合為7，因為advanced的output dimension是7筆binary classification的資料。
optimizer用adam，讓learning rate 可以更加平穩，因為他會保留過去的momentum和矯正 偏離，loss用binary_crossentropy，因為這是binary classification問題。
有些問題與basic差不多，但一開始做成multiple classificaiton，變成分類文題，造成7個label 只有1個是1其他是0，後來才了解是要用7個binary classification來輸出output。 
1.原本遇到loss太⾼高，accuracy太低的問題，最後使⽤用Drop out有改善許多。
2.要將input size 做Reshape成(10000,128,128,1)，加上最後⼀一個顏⾊色的參參數，不然size會不合沒辦法訓練模型。
3.loss要⽤用.binary Crossentropy⽽而不是catagory，因為這是binary classification不是multiple classification問題，是判斷單⼀一label是0還是1⽽而不是分類問題，label們都有可能為0或1。 4.Dense number也是印出來來看了了很多次確認最後有沒有輸出正確的output size。
5.activate function⽤用Sigmoid跟softmax的結果差很多，sigmoid才是適合binary classification。
根據以上注意事項將x_train y_train reshape調整input size後輸入神經網路模型中萃取圖片資
料，訓練後再將test資料輸入模型得到output預測結果並印出csv檔案。這部分跟basic不同的 是輸出的維度為7，有7個label。
## 3. Decision Tree
## Binary Entropy Classifier 
![Variable Declaration](/mlimg/tree_visualization.png)
## Report
MIMIC data 預測 model:
Top 3 splitting features and their thresholds:
1. mvar12 <= 0.5 
2. CMO <= 0.5 
3. mvar23 <=0.5
Build decision tree:
1. train_test_split:分出x_train, y_train, x_test, y_test
2.Import decision tree 跟 random forest 
3.將x_train, y_train 輸入model訓練
4.拿分好的y_test跟prediction結果分別算出accuracy
5.import x_test 得到預測的 y_pred
## 4. Q Reinforcement Learning
## Reinforcement Learning from QTable and Deep Q Network
## Cartpole Visulization
![Variable Declaration](/mlimg/cartpole.gif)
![Variable Declaration](/mlimg/qlearning.png)
## Report
* Basic:
先在net中建立神經網路路，包含隱藏層，並得到action的分數，第⼆步再dqn中則是建立q learning network。重要的部分兩兩個net:eval_net及target_net，其他為設定self的各個參參數值以及設定memory 保存⼤大⼩小。在choose_action中隨機學習經驗與選擇最⾼高分的action，⽽而後在store_transition中store experience。將每回的reward加起來來後，進⾏行行learn()訓練。⽽而這裡的reward有修改過，根據柱⼦子的 radius分配更更⼤大的reward，這樣的訓練效果比原本好很多。
在調參數時遇到蠻多困難的，⾸先N_EPISODES很直觀，但有點搞不懂EPISODE_LENGTH有什麼作用，後來才知道那會影響每⼀回合的step。還有epispdes要設到3000以上效果才會比較好，設4000 跑了很久，但是更更穩定，test也成功過關。
* Advanced - 將state-action pair存到q table中，直接從裡⾯面拿資料與更更新資料
這裡的做法是將state-action pair存到q table中，直接從裡⾯面拿資料與更更新資料。在choose_action 中⽤用隨機機率學習經驗與選擇最⾼高分的action，在get_state終將連續特徵轉為離散，⽤用bucket表 ⽰示，⽽而後在下⾯面調整bucket的參參數進⾏行行訓練。列列出state範圍個參參數後，將state-action pair存到q table中，進⾏行行訓練，⼀一樣會累積reward，⽽而這裡⽤用的公式就是q learning公式，計算完成後存到 table中再繼續訓練。
在設定state-action pair遇到蠻多困難的，試了了很久才想到要⽤用(action,)的形式，還有bucket的調 參參，第三個feature從1慢慢開始往上加訓練，發現只有6才能訓練成功。
## 5. Stockprice Prediction
## Basic method of Linear Regression 
## Advanced method of LSTM
![Variable Declaration](/mlimg/linear.jpeg)
## Report
* Basic prediction: *
Regression的部分我兩兩種OLS,gradient都有做並分別去train，再選lost必較⼩小的model來來實際丟資料 進去做預測，最後我選擇OLS因為loss較⼩小。 ⼀一開始splitdata，我也分別測試了了不同資料數量量丟下去train會得到的lost的不同，發現⽤用10/14號前 60天的資料訓練比只⽤用最後30筆或⽤用到最後90筆的效能還要更更好，所以最後決定⽤用 input_datalist[129:189]#前60天 來來做，最後再⽤用前20天的資料當validation來來測試。
Ordinary Least Square Solution:
先將我引入的x data轉成numpy形式，並添加⼀一⾏行行都是1的vector，因為w會有k+1項，必須要符合那
個shape，同時把x與y都增加⼀一維，接著帶入Ordinary Least Square Solution公式計算出我linear regression的weight與bias。
再將我準備好的validation帶入w與b並⽤用plt做圖來來看我的valx,valy點，也可以清楚的觀察測試資料 valx對應到的預測y值以及求出的linear function。
Gradient descent:
⼀一樣輸入測試集x,y，⽤用gradient descent的算式去重複做，先算x與w矩陣相乘與實際y值的差(lost) ，並求微分過後的error function，再跟⾃自訂的learning rate相乘，⼀一次⼀一次的修正theta直到找到使 error最⼩小值的w所在。經過不斷地調參參與測試loss值，我決定了了learning_rate=0.0000000001， gradient descent的次數總共為1000次，loss是我觀察到的最⼩小值。 最後⼀一樣，再將我準備好的validation帶入w與b並⽤用plt做圖來來看我的valx,valy點，可以清楚的觀察測 試資料valx對應到的預測y值以及求出的linear function。
最後將我10/15後⼆二⼗十天的stock price作為x帶入function得到輸出結果。
Training data 數量量與loss function算出的loss比較: 
根據實際測試，最後選擇⽤用loss最⼩小的OLS Regression以及60筆資料來來訓練。
![Variable Declaration](/mlimg/compare.png)
* Advanced prediction:
至於advanced的部分，我使用的是Keras的LSTM model。用TSMC每筆資料的前80天到20天的資料來訓練這⼀天的股價，最後實際使⽤時輸入前60天股價來預測下⼀天股價。第⼀步我先用MinMaxScaler做正規化到01之間，避免資料過⼤，再做numpy、 xdata的reshape處理，變成每60筆資料為一份資料集來訓練我的LSTM model。 LSTM我用128跟64⼀共⽤了兩層，中間有特別用Dropout(0.2)增加效能。 得到model之後，我將所有資料輸入model得到對應的預測y，再跟⾃己做root mean square跟mape算出loss:
Rsme = 14.905772239305383 
Mape = 1.998070466413709
可以觀察到時間序列列預測的效能明顯比basic的線性regression好很多。 接下來來實際⽤用model來來預測，⽤用iteration⼀一次丟i的前60天股價預測出第i天股價，再將預測出的股價 寫回testdata，作為下⼀一天股價的60筆預測資料中最後⼀一個最新的資料，序列列式的做完最後20天的 股價預測。
## Final Result
![Variable Declaration](/mlimg/stock_price_prediction_2022.png)
