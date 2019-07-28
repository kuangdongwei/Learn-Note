
# 建立多层感知器进行IMDb情感分析

## 读取数据 


```python
import os
import re 
'''删除文字中的HTML标签'''
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

'''读取IMDB文件目录'''
def read_files(filetype):
    path = 'datas/aclImdb/'
    file_list = []
    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
        
    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
        
    print('read ',filetype,len(file_list),'files')
    
    all_labels = ([1]*12500 + [0]*12500)
    all_text = []
    for fi in file_list:
        with open(fi,encoding='utf-8') as finput:
            all_text += [rm_tags(" ".join(finput.readlines()))]
            
    return all_labels,all_text
```

## 数据预处理


```python
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
#读取训练数据
y_train,train_text = read_files("train")
#读取测试数据
y_test,test_text = read_files("test")
#建立token
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)
#将影评文字转换为数字列表
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
#将所有数字列表统一长度为100
x_train = sequence.pad_sequences(x_train_seq,maxlen=100)
x_test = sequence.pad_sequences(x_train_seq,maxlen=100)
```

    Using TensorFlow backend.
    

    read  train 25000 files
    read  test 25000 files
    

## 加入嵌入层


```python
#导入所需要的模块
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding

model = Sequential()
#输出为32维，输入为2000维，每一项影评的长度为100词
model.add(Embedding(output_dim = 32 ,
                    input_dim = 2000,
                    input_length = 100))
#加入dropout避免过拟合
model.add(Dropout(0.2))
```

    WARNING:tensorflow:From E:\Myanaconda\anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From E:\Myanaconda\anaconda\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    

## 建立多层感知器模型


```python
#加入平坦层
model.add(Flatten())
```


```python
#加入隐藏层
model.add(Dense(units = 256,
                activation = 'relu'))
model.add(Dropout(0.35))
#加入输出层
model.add(Dense(units=1,
                activation='sigmoid'))
```


```python
#查看模型摘要信息
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 32)           64000     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 100, 32)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3200)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               819456    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 883,713
    Trainable params: 883,713
    Non-trainable params: 0
    _________________________________________________________________
    

## 训练模型


```python
#定义训练方式
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#开始训练
train_history = model.fit(x_train,y_train,batch_size=100,
                          epochs=10,
                          verbose=0.2,
                          validation_split=0.2)
```

    WARNING:tensorflow:From E:\Myanaconda\anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
    Epoch 2/10
    Epoch 3/10
    Epoch 4/10
    Epoch 5/10
    Epoch 6/10
    Epoch 7/10
    Epoch 8/10
    Epoch 9/10
    Epoch 10/10
    

## 模型准确率评估及预测


```python
#评估模型准确率
scores = model.evaluate(x_test,y_test,verbose=1)
```

    25000/25000 [==============================] - 3s 108us/step
    


```python
#进行预测
predict = model.predict_classes(x_test)
```


```python
#查看预测结果的前10项
predict[:10]
```




    array([[1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1]])




```python
#将上面二维结果转换成一维
predict_classes = predict.reshape(-1)
predict_classes[:10]
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
#创建一个函数用来查看测试数据预测结果
sentimentDict = {1:'正面评价',0:'负面评价'}
def display_test_sentiment(i):
    #显示测试数据的第i条影评
    print(test_text[i])
    print('label真实值',sentimentDict[y_test[i]],'预测结果',sentimentDict[predict_classes[i]])
```


```python
#查看测试数据的第10条影评的预测结果
display_test_sentiment(10)
print(' ')
#查看测试数据的第20000条影评的预测结果
display_test_sentiment(20000)
```

    I loved this movie from beginning to end.I am a musician and i let drugs get in the way of my some of the things i used to love(skateboarding,drawing) but my friends were always there for me.Music was like my rehab,life support,and my drug.It changed my life.I can totally relate to this movie and i wish there was more i could say.This movie left me speechless to be honest.I just saw it on the Ifc channel.I usually hate having satellite but this was a perk of having satellite.The ifc channel shows some really great movies and without it I never would have found this movie.Im not a big fan of the international films because i find that a lot of the don't do a very good job on translating lines.I mean the obvious language barrier leaves you to just believe thats what they are saying but its not that big of a deal i guess.I almost never got to see this AMAZING movie.Good thing i stayed up for it instead of going to bed..well earlier than usual.lol.I hope you all enjoy the hell of this movie and Love this movie just as much as i did.I wish i could type this all in caps but its again the rules i guess thats shouting but it would really show my excitement for the film.I Give It Three Thumbs Way Up!This Movie Blew ME AWAY!
    label真实值 正面评价 预测结果 正面评价
     
    this film had a lot of potential - it's a great story and has the potential to be very creepy. but of course tim burton doesn't really do creepy films, he does wacky cartoonish films. and i usually like tim burton's stuff. but i thought this film was really weak. the best thing about the film (and it is actually worth seeing just for this) was the art direction - the film has an amazing intangible quality to it. the script was not good. it was boring in parts and confusing in other parts, and there was no building of characters. i never really cared that people were having their heads lopped off by a headless being. i thought johnny depp had a good thing going with his approach to the character, but given that the script was weak he couldn't go too far with it - and i was very irritated by the attempts at a slight accent on his and christina ricci's parts.anyway, it is sadly not a great film and not worth seeing unless you are interested in the art direction.
    label真实值 负面评价 预测结果 负面评价
    

## 查看《美女与野兽的影评》


```python
#创建一个函数用于对需要预测的影评文本进行处理
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=100)
    predict_result = model.predict_classes(pad_input_seq)
    predict = predict_result.reshape(-1)
    print('预测文本:',input_text)
    print('预测结果:',predict[0])
```


```python
coment_text = '''Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.'''

predict_review(coment_text)
```

    预测文本: Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
    Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
    from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
    makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
    animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
    between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
    because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.
    预测结果: 0
    

# 使用Keras RNN和LSTM进行IMDb情感分析

## 使用Keras RNN分析


```python
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model_rnn = Sequential()
#加入嵌入层
model_rnn.add(Embedding(output_dim = 32,
                        input_dim = 2000,
                        input_length = 100))
#加入RNN网络
model_rnn.add(SimpleRNN(units=16))
#加入隐藏层
model_rnn.add(Dense(units=256,
                    activation='relu'))
model_rnn.add(Dropout(0.35))
#加入输出层
model_rnn.add(Dense(units=1,
              activation='sigmoid'))
```


```python
model_rnn.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 100, 32)           64000     
    _________________________________________________________________
    simple_rnn_2 (SimpleRNN)     (None, 16)                784       
    _________________________________________________________________
    dense_5 (Dense)              (None, 256)               4352      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 69,393
    Trainable params: 69,393
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model_rnn.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
```


```python
rnn_train_history = model.fit(x_train,y_train,batch_size=100,
                          epochs=8,
                          verbose=0.2,
                          validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/8
    Epoch 2/8
    Epoch 3/8
    Epoch 4/8
    Epoch 5/8
    Epoch 6/8
    Epoch 7/8
    Epoch 8/8
    


```python
model_rnn.evaluate(x_test,y_test,verbose=1)
```

    25000/25000 [==============================] - 6s 259us/step
    




    [0.6931595656204224, 0.50388]




```python
predict_rnn = model_rnn.predict_classes(x_test)
```


```python
predict_rnn_result = predict_rnn.reshape(-1)
predict_rnn_result[10000]
```




    1




```python
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=100)
    predict_result = model_rnn.predict_classes(pad_input_seq)
    predict = predict_result.reshape(-1)
    print('预测文本:',input_text)
    print('预测结果:',predict[0])
    
coment_text2 = '''Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.'''

predict_review(coment_text2)
```

    预测文本: Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
    Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
    from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
    makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
    animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
    between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
    because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.
    预测结果: 1
    

问题：在模型测试就可以看到其准确率只有百分之五六十，效果比多层感知器模型还要差。刚刚我们对多层感知器中使用过的一段影评做统一的预测在这里得到的结果是1（正面评价），而多层感知器模型中得到的结果是0（负面的）。很容易读出这段影评肯定是负面的，因此RNN模型在这的效果不如多层感知器模型，按理来说RNN模型对于这种预测应该得到更好的效果

原因：评价字段的最大长度不够，导致对于长文本评价丢失了许多信息；可能训练批次相比于多层感知器中减少了两次带来了影响；在嵌入层下面没有加入dropout避免过拟合


```python
#保存模型，以便下次直接使用，而不需要重新训练
model_rnn.save('models/RNN-sentiment')
```

## 使用keras LSTM分析


```python
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
```


```python
model_lstm = Sequential()
#嵌入层
model_lstm.add(Embedding(output_dim = 32,
                         input_dim = 2000,
                         input_length=100))
model_lstm.add(Dropout(0.2))
#加入LSTM网络
model_lstm.add(LSTM(32))
#隐藏层和输出层
model_lstm.add(Dense(units=256,
                     activation='relu'))
model_lstm.add(Dropout(0.35))
model_lstm.add(Dense(units=1,
                     activation='sigmoid'))
```


```python
model_lstm.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 100, 32)           64000     
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 100, 32)           0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 32)                8320      
    _________________________________________________________________
    dense_7 (Dense)              (None, 256)               8448      
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 81,025
    Trainable params: 81,025
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#定义训练方式
model_lstm.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
```


```python
#开始训练
train_history_lstm = model_lstm.fit(x_train,y_train,batch_size=100,
                                    epochs=10,
                                    verbose=2,
                                    validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
     - 29s - loss: 0.4929 - acc: 0.7516 - val_loss: 0.5737 - val_acc: 0.7438
    Epoch 2/10
     - 26s - loss: 0.3210 - acc: 0.8624 - val_loss: 0.5977 - val_acc: 0.7578
    Epoch 3/10
     - 24s - loss: 0.2997 - acc: 0.8762 - val_loss: 0.5564 - val_acc: 0.7616
    Epoch 4/10
     - 24s - loss: 0.2825 - acc: 0.8833 - val_loss: 0.3738 - val_acc: 0.8238
    Epoch 5/10
     - 25s - loss: 0.2700 - acc: 0.8887 - val_loss: 0.3783 - val_acc: 0.8374
    Epoch 6/10
     - 25s - loss: 0.2552 - acc: 0.8962 - val_loss: 0.3483 - val_acc: 0.8400
    Epoch 7/10
     - 26s - loss: 0.2431 - acc: 0.9002 - val_loss: 0.4039 - val_acc: 0.8128
    Epoch 8/10
     - 29s - loss: 0.2339 - acc: 0.9024 - val_loss: 0.5755 - val_acc: 0.7504
    Epoch 9/10
     - 30s - loss: 0.2196 - acc: 0.9124 - val_loss: 0.3386 - val_acc: 0.8560
    Epoch 10/10
     - 29s - loss: 0.2110 - acc: 0.9155 - val_loss: 0.6526 - val_acc: 0.7724
    


```python
#模型准确率评估
model_lstm.evaluate(x_test,y_test,verbose=1)
```

    25000/25000 [==============================] - 12s 473us/step
    




    [0.26547247427463533, 0.90276]




```python
#对测试数据进行预测
predict_lstm = model_lstm.predict_classes(x_test,batch_size=100,verbose=1)
```

    25000/25000 [==============================] - 14s 574us/step
    


```python
predict_lstm_result = predict_lstm.reshape(-1)
#任意查看几条预测数据
predict_lstm_result[10000:10006]
```




    array([1, 1, 1, 1, 1, 1])




```python
display_test_sentiment(12500)
```

    Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.
    label真实值 负面评价 预测结果 负面评价
    


```python
#同样的还是对那一段《美女与野兽》影评进行分析预测
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=100)
    predict_result = model_lstm.predict_classes(pad_input_seq)
    predict = predict_result.reshape(-1)
    print('预测文本:',input_text)
    print('预测结果:',predict[0])
    
coment_text2 = '''Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.'''

predict_review(coment_text2)
```

    预测文本: Where do I start. This adaptation of Disney's 1991 Beauty and the Beast was an utter disappointment. 
    Emma Watson as Belle was extremely unconvincing from the start to the end. She had the same expressions as the actress
    from Twilight. The animators did a terrible job with the Beast. He looked fake and lifeless. They could have used special
    makeup to create the beast similar to the Grinch where we get to see Jim Carrey's expressions. The side character
    animations were poorly executed. Overall I felt the film was rushed as there was lack of compassion and chemistry
    between the characters. There was a lot of CGI and green screen which could have been replaced by normal acting,
    because then why make an animated version of an animated film? This is by far the worst remake of an animated classic.
    预测结果: 0
    

可以看到LSTM模型的准确率达到了90%左右，预测结果也是很不错的，不愧为NLP界的最强模型


```python

```
