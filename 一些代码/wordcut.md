{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['南京大学------', '占领------', '了------', '南京市------', '长江大桥------']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "中文分词技术-规则分词\n",
    "  正向最大匹配法\n",
    "'''\n",
    "class MM(object):\n",
    "    def __init__(self):\n",
    "        self.window_size = 4\n",
    "    def cut(self,text):\n",
    "        result =[]\n",
    "        index = 0\n",
    "        text_length = len(text)\n",
    "        dic = ['南京市','长江大桥','南京大学','占领']\n",
    "        while text_length > index:\n",
    "            for size in range(self.window_size+index,index,-1):\n",
    "                piece = text[index:size]\n",
    "                if piece in dic:\n",
    "                    index = size - 1 \n",
    "                    break\n",
    "            index = index + 1 \n",
    "            result.append(piece+'------')\n",
    "        print(result)\n",
    "        \n",
    "if __name__ == '__main__':\n",
    "    text = '南京大学占领了南京市长江大桥'\n",
    "    tokenizer = MM()\n",
    "    print(tokenizer.cut(text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['研究/ ', '生命/ ', '的/ ', '起源/ ']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "中文分词技术-规则分词\n",
    "  逆向最大匹配法\n",
    "'''\n",
    "class RMM(object):\n",
    "    def __init__(self):\n",
    "        self.window_size=3 \n",
    "    \n",
    "    def cut(self,txt):\n",
    "        result = []\n",
    "        index = len(txt)\n",
    "        dic = ['研究','研究生','生命','的','起源']\n",
    "        while index > 0:\n",
    "            for size in range(index - self.window_size,index):\n",
    "                piece = txt[size:index]\n",
    "                if piece in dic:\n",
    "                    index = size + 1 \n",
    "                    break\n",
    "            index = index-1 \n",
    "            result.append(piece+'/ ')\n",
    "        result.reverse()\n",
    "        print(result)\n",
    "        \n",
    "if __name__ == '__main__':\n",
    "    text = '研究生命的起源'\n",
    "    tokenizers = RMM()\n",
    "    print(tokenizers.cut(text))\n",
    "          "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "统计分词法--HMM模型\n",
    "'''\n",
    "class HMM(object):\n",
    "    def __init__(self):\n",
    "        import os\n",
    "\n",
    "        # 主要是用于存取算法中间结果，不用每次都训练模型\n",
    "        self.model_file = 'models/hmm_model.pkl'\n",
    "\n",
    "        # 状态值集合\n",
    "        self.state_list = ['B', 'M', 'E', 'S']\n",
    "        # 参数加载,用于判断是否需要重新加载model_file\n",
    "        self.load_para = False\n",
    "\n",
    "    # 用于加载已计算的中间结果，当需要重新训练时，需初始化清空结果\n",
    "    def try_load_model(self, trained):\n",
    "        if trained:\n",
    "            import pickle\n",
    "            with open(self.model_file, 'rb') as f:\n",
    "                self.A_dic = pickle.load(f)\n",
    "                self.B_dic = pickle.load(f)\n",
    "                self.Pi_dic = pickle.load(f)\n",
    "                self.load_para = True\n",
    "\n",
    "        else:\n",
    "            # 状态转移概率（状态->状态的条件概率）\n",
    "            self.A_dic = {}\n",
    "            # 发射概率（状态->词语的条件概率）\n",
    "            self.B_dic = {}\n",
    "            # 状态的初始概率\n",
    "            self.Pi_dic = {}\n",
    "            self.load_para = False\n",
    "\n",
    "    # 计算转移概率、发射概率以及初始概率\n",
    "    def train(self, path):\n",
    "\n",
    "        # 重置几个概率矩阵\n",
    "        self.try_load_model(False)\n",
    "\n",
    "        # 统计状态出现次数，求p(o)\n",
    "        Count_dic = {}\n",
    "\n",
    "        # 初始化参数\n",
    "        def init_parameters():\n",
    "            for state in self.state_list:\n",
    "                self.A_dic[state] = {s: 0.0 for s in self.state_list}\n",
    "                self.Pi_dic[state] = 0.0\n",
    "                self.B_dic[state] = {}\n",
    "\n",
    "                Count_dic[state] = 0\n",
    "\n",
    "        def makeLabel(text):\n",
    "            out_text = []\n",
    "            if len(text) == 1:\n",
    "                out_text.append('S')\n",
    "            else:\n",
    "                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']\n",
    "\n",
    "            return out_text\n",
    "\n",
    "        init_parameters()\n",
    "        line_num = -1\n",
    "        # 观察者集合，主要是字以及标点等\n",
    "        words = set()\n",
    "        with open(path, encoding='utf8') as f:\n",
    "            for line in f:\n",
    "                line_num += 1\n",
    "\n",
    "                line = line.strip()\n",
    "                if not line:\n",
    "                    continue\n",
    "\n",
    "                word_list = [i for i in line if i != ' ']\n",
    "                words |= set(word_list)  # 更新字的集合\n",
    "\n",
    "                linelist = line.split()\n",
    "\n",
    "                line_state = []\n",
    "                for w in linelist:\n",
    "                    line_state.extend(makeLabel(w))\n",
    "                \n",
    "                assert len(word_list) == len(line_state)\n",
    "\n",
    "                for k, v in enumerate(line_state):\n",
    "                    Count_dic[v] += 1\n",
    "                    if k == 0:\n",
    "                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率\n",
    "                    else:\n",
    "                        self.A_dic[line_state[k - 1]][v] += 1  # 计算转移概率\n",
    "                        self.B_dic[line_state[k]][word_list[k]] = \\\n",
    "                            self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0  # 计算发射概率\n",
    "        \n",
    "        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}\n",
    "        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()}\n",
    "                      for k, v in self.A_dic.items()}\n",
    "        #加1平滑\n",
    "        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()}\n",
    "                      for k, v in self.B_dic.items()}\n",
    "        #序列化\n",
    "        import pickle\n",
    "        with open(self.model_file, 'wb') as f:\n",
    "            pickle.dump(self.A_dic, f)\n",
    "            pickle.dump(self.B_dic, f)\n",
    "            pickle.dump(self.Pi_dic, f)\n",
    "\n",
    "        return self\n",
    "\n",
    "    def viterbi(self, text, states, start_p, trans_p, emit_p):\n",
    "        V = [{}]\n",
    "        path = {}\n",
    "        for y in states:\n",
    "            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)\n",
    "            path[y] = [y]\n",
    "        for t in range(1, len(text)):\n",
    "            V.append({})\n",
    "            newpath = {}\n",
    "            \n",
    "            #检验训练的发射概率矩阵中是否有该字\n",
    "            neverSeen = text[t] not in emit_p['S'].keys() and \\\n",
    "                text[t] not in emit_p['M'].keys() and \\\n",
    "                text[t] not in emit_p['E'].keys() and \\\n",
    "                text[t] not in emit_p['B'].keys()\n",
    "            for y in states:\n",
    "                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0 #设置未知字单独成词\n",
    "                (prob, state) = max(\n",
    "                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) *\n",
    "                      emitP, y0)\n",
    "                     for y0 in states if V[t - 1][y0] > 0])\n",
    "                V[t][y] = prob\n",
    "                newpath[y] = path[state] + [y]\n",
    "            path = newpath\n",
    "            \n",
    "        if emit_p['M'].get(text[-1], 0)> emit_p['S'].get(text[-1], 0):\n",
    "            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E','M')])\n",
    "        else:\n",
    "            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])\n",
    "        \n",
    "        return (prob, path[state])\n",
    "\n",
    "    def cut(self, text):\n",
    "        import os\n",
    "        if not self.load_para:\n",
    "            self.try_load_model(os.path.exists(self.model_file))\n",
    "        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)      \n",
    "        begin, next = 0, 0    \n",
    "        for i, char in enumerate(text):\n",
    "            pos = pos_list[i]\n",
    "            if pos == 'B':\n",
    "                begin = i\n",
    "            elif pos == 'E':\n",
    "                yield text[begin: i+1]\n",
    "                next = i+1\n",
    "            elif pos == 'S':\n",
    "                yield char\n",
    "                next = i+1\n",
    "        if next < len(text):\n",
    "            yield text[next:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<__main__.HMM at 0x65d5dd8>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hmm = HMM()\n",
    "hmm.train('datas/trainCorpus.txt_utf8')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['分词', '技术', '和', '命名', '实体', '识别', '是', '自然', '语言', '处理', '中', '非常', '重要', '的', '基础', '技术', '，', '也', '是', '构建', '知识', '图', '谱', '过程', '中', '不可', '或', '缺', '的', '步骤']\n"
     ]
    }
   ],
   "source": [
    "text = '分词技术和命名实体识别是自然语言处理中非常重要的基础技术，也是构建知识图谱过程中不可或缺的步骤'\n",
    "res = hmm.cut(text)\n",
    "print(str(list(res)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "全模式: 中文/ 分词/ 是/ 自然/ 自然语言/ 语言/ 处理/ 不可/ 不可或缺/ 或缺/ 的/ 重要/ 的/ 步骤\n",
      "精确模式 中文/ 分词/ 是/ 自然语言/ 处理/ 不可或缺/ 的/ 重要/ 的/ 步骤\n",
      "搜索模式  中文/ 分词/ 是/ 自然/ 语言/ 自然语言/ 处理/ 不可/ 或缺/ 不可或缺/ 的/ 重要/ 的/ 步骤\n"
     ]
    }
   ],
   "source": [
    "#结巴分词三种模式对比\n",
    "import jieba\n",
    "sent = '中文分词是自然语言处理不可或缺的重要的步骤'\n",
    "seg_list = jieba.cut(sent,cut_all=True)\n",
    "print('全模式:','/ '.join(seg_list))\n",
    "seg_list = jieba.cut(sent,cut_all=False)\n",
    "print('精确模式','/ '.join(seg_list))\n",
    "seg_list = jieba.cut_for_search(sent)\n",
    "print('搜索模式 ','/ '.join(seg_list))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "can only concatenate str (not \"int\") to str",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-59e8fc444f9b>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     29\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     30\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'__main__'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 31\u001b[1;33m     \u001b[0mmain\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-1-59e8fc444f9b>\u001b[0m in \u001b[0;36mmain\u001b[1;34m()\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     21\u001b[0m     \u001b[0mfiles\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mglob\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mglob\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'datas/newsc000013/*.txt'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 22\u001b[1;33m     \u001b[0mcorpus\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mget_content\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mfiles\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     23\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     24\u001b[0m     \u001b[0msample_inx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mrandom\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrandint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-1-59e8fc444f9b>\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     21\u001b[0m     \u001b[0mfiles\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mglob\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mglob\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'datas/newsc000013/*.txt'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 22\u001b[1;33m     \u001b[0mcorpus\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mget_content\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mfiles\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     23\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     24\u001b[0m     \u001b[0msample_inx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mrandom\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrandint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-1-59e8fc444f9b>\u001b[0m in \u001b[0;36mget_content\u001b[1;34m(path)\u001b[0m\n\u001b[0;32m      5\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0ml\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m             \u001b[0ml\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0ml\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstrip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m             \u001b[0mcontent\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      8\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mcontent\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: can only concatenate str (not \"int\") to str"
     ]
    }
   ],
   "source": [
    "'''实战之高频词提取'''\n",
    "def get_content(path):\n",
    "    with open(path,'r',encoding='gbk',errors='ignore') as f:\n",
    "        content = ''\n",
    "        for l in f:\n",
    "            l = l.strip()\n",
    "            content += 1\n",
    "        return content\n",
    "    \n",
    "def get_tf(words,topk=10):\n",
    "    tf_dic = {}\n",
    "    for w in words:\n",
    "        tf_dic[w]=tf_dic.get(w,0) + 1 \n",
    "    return sorted(tf_dic.items(),key=lambda x: x[1],reverse=True) [:topk]\n",
    "\n",
    "def main():\n",
    "    import glob\n",
    "    import random\n",
    "    import jieba\n",
    "    \n",
    "    files = glob.glob('datas/newsc000013/*.txt')\n",
    "    corpus = [get_content(x) for x in files]\n",
    "    \n",
    "    sample_inx = random.randint(0,len(corpus))\n",
    "    split_words = list(jieba.cut(corpus[sample_inx]))\n",
    "    print('样本之一: '+corpus[sample_inx])\n",
    "    print('样本分词效果: '+'/ '.join(split_words))\n",
    "    print('样本的topk(10)词 ： '+str(get_tf(split_words)))\n",
    "    \n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "他 骑 自行车 去 市场 买 了 好多 的 菜 , 准备 今晚 吃 大餐\n"
     ]
    }
   ],
   "source": [
    "import jieba\n",
    "string = '他骑自行车去市场买了好多的菜,准备今晚吃大餐'\n",
    "seg_list = jieba.cut(string,cut_all=False)\n",
    "seg_result = ' '.join(seg_list)\n",
    "print(seg_result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "generator"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(seg_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
