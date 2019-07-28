

```python
'''
中文分词技术-规则分词
  正向最大匹配法
'''
class MM(object):
    def __init__(self):
        self.window_size = 4
    def cut(self,text):
        result =[]
        index = 0
        text_length = len(text)
        dic = ['南京市','长江大桥','南京大学','占领']
        while text_length > index:
            for size in range(self.window_size+index,index,-1):
                piece = text[index:size]
                if piece in dic:
                    index = size - 1 
                    break
            index = index + 1 
            result.append(piece+'------')
        print(result)
        
if __name__ == '__main__':
    text = '南京大学占领了南京市长江大桥'
    tokenizer = MM()
    print(tokenizer.cut(text))
```

    ['南京大学------', '占领------', '了------', '南京市------', '长江大桥------']
    None
    


```python
'''
中文分词技术-规则分词
  逆向最大匹配法
'''
class RMM(object):
    def __init__(self):
        self.window_size=3 
    
    def cut(self,txt):
        result = []
        index = len(txt)
        dic = ['研究','研究生','生命','的','起源']
        while index > 0:
            for size in range(index - self.window_size,index):
                piece = txt[size:index]
                if piece in dic:
                    index = size + 1 
                    break
            index = index-1 
            result.append(piece+'/ ')
        result.reverse()
        print(result)
        
if __name__ == '__main__':
    text = '研究生命的起源'
    tokenizers = RMM()
    print(tokenizers.cut(text))
          
```

    ['研究/ ', '生命/ ', '的/ ', '起源/ ']
    None
    


```python
'''
统计分词法--HMM模型
'''
class HMM(object):
    def __init__(self):
        import os

        # 主要是用于存取算法中间结果，不用每次都训练模型
        self.model_file = 'models/hmm_model.pkl'

        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        # 参数加载,用于判断是否需要重新加载model_file
        self.load_para = False

    # 用于加载已计算的中间结果，当需要重新训练时，需初始化清空结果
    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True

        else:
            # 状态转移概率（状态->状态的条件概率）
            self.A_dic = {}
            # 发射概率（状态->词语的条件概率）
            self.B_dic = {}
            # 状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    # 计算转移概率、发射概率以及初始概率
    def train(self, path):

        # 重置几个概率矩阵
        self.try_load_model(False)

        # 统计状态出现次数，求p(o)
        Count_dic = {}

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                self.Pi_dic[state] = 0.0
                self.B_dic[state] = {}

                Count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

            return out_text

        init_parameters()
        line_num = -1
        # 观察者集合，主要是字以及标点等
        words = set()
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                word_list = [i for i in line if i != ' ']
                words |= set(word_list)  # 更新字的集合

                linelist = line.split()

                line_state = []
                for w in linelist:
                    line_state.extend(makeLabel(w))
                
                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率
                    else:
                        self.A_dic[line_state[k - 1]][v] += 1  # 计算转移概率
                        self.B_dic[line_state[k]][word_list[k]] = \
                            self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0  # 计算发射概率
        
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.A_dic.items()}
        #加1平滑
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.B_dic.items()}
        #序列化
        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            
            #检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0 #设置未知字单独成词
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) *
                      emitP, y0)
                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
            
        if emit_p['M'].get(text[-1], 0)> emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E','M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        
        return (prob, path[state])

    def cut(self, text):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)      
        begin, next = 0, 0    
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next = i+1
            elif pos == 'S':
                yield char
                next = i+1
        if next < len(text):
            yield text[next:]
```


```python
hmm = HMM()
hmm.train('datas/trainCorpus.txt_utf8')

```




    <__main__.HMM at 0x65d5dd8>




```python
text = '分词技术和命名实体识别是自然语言处理中非常重要的基础技术，也是构建知识图谱过程中不可或缺的步骤'
res = hmm.cut(text)
print(str(list(res)))
```

    ['分词', '技术', '和', '命名', '实体', '识别', '是', '自然', '语言', '处理', '中', '非常', '重要', '的', '基础', '技术', '，', '也', '是', '构建', '知识', '图', '谱', '过程', '中', '不可', '或', '缺', '的', '步骤']
    


```python
#结巴分词三种模式对比
import jieba
sent = '中文分词是自然语言处理不可或缺的重要的步骤'
seg_list = jieba.cut(sent,cut_all=True)
print('全模式:','/ '.join(seg_list))
seg_list = jieba.cut(sent,cut_all=False)
print('精确模式','/ '.join(seg_list))
seg_list = jieba.cut_for_search(sent)
print('搜索模式 ','/ '.join(seg_list))
```

    全模式: 中文/ 分词/ 是/ 自然/ 自然语言/ 语言/ 处理/ 不可/ 不可或缺/ 或缺/ 的/ 重要/ 的/ 步骤
    精确模式 中文/ 分词/ 是/ 自然语言/ 处理/ 不可或缺/ 的/ 重要/ 的/ 步骤
    搜索模式  中文/ 分词/ 是/ 自然/ 语言/ 自然语言/ 处理/ 不可/ 或缺/ 不可或缺/ 的/ 重要/ 的/ 步骤
    


```python
'''实战之高频词提取'''
def get_content(path):
    with open(path,'r',encoding='gbk',errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += 1
        return content
    
def get_tf(words,topk=10):
    tf_dic = {}
    for w in words:
        tf_dic[w]=tf_dic.get(w,0) + 1 
    return sorted(tf_dic.items(),key=lambda x: x[1],reverse=True) [:topk]

def main():
    import glob
    import random
    import jieba
    
    files = glob.glob('datas/newsc000013/*.txt')
    corpus = [get_content(x) for x in files]
    
    sample_inx = random.randint(0,len(corpus))
    split_words = list(jieba.cut(corpus[sample_inx]))
    print('样本之一: '+corpus[sample_inx])
    print('样本分词效果: '+'/ '.join(split_words))
    print('样本的topk(10)词 ： '+str(get_tf(split_words)))
    
if __name__ == '__main__':
    main()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-1-59e8fc444f9b> in <module>
         29 
         30 if __name__ == '__main__':
    ---> 31     main()
    

    <ipython-input-1-59e8fc444f9b> in main()
         20 
         21     files = glob.glob('datas/newsc000013/*.txt')
    ---> 22     corpus = [get_content(x) for x in files]
         23 
         24     sample_inx = random.randint(0,len(corpus))
    

    <ipython-input-1-59e8fc444f9b> in <listcomp>(.0)
         20 
         21     files = glob.glob('datas/newsc000013/*.txt')
    ---> 22     corpus = [get_content(x) for x in files]
         23 
         24     sample_inx = random.randint(0,len(corpus))
    

    <ipython-input-1-59e8fc444f9b> in get_content(path)
          5         for l in f:
          6             l = l.strip()
    ----> 7             content += 1
          8         return content
          9 
    

    TypeError: can only concatenate str (not "int") to str



```python
import jieba
string = '他骑自行车去市场买了好多的菜,准备今晚吃大餐'
seg_list = jieba.cut(string,cut_all=False)
seg_result = ' '.join(seg_list)
print(seg_result)
```

    他 骑 自行车 去 市场 买 了 好多 的 菜 , 准备 今晚 吃 大餐
    


```python
type(seg_list)
```




    generator




```python

```
