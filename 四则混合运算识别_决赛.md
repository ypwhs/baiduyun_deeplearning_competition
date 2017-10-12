# 四则混合运算识别（决赛）
 
本节会详细介绍我在进行四则混合运算识别竞赛决赛时的所有思路。

## 问题描述

本次竞赛目的是为了解决一个 OCR 问题，通俗地讲就是实现图像到文字的转换过程。

### 数据集

决赛数据集一共包含10万张图片和一个labels.txt的文本文件。每张图片包含一个数学运算式，运算式中包含：

1. 图片大小不固定
2. 图片中的某一块区域为公式部分
3. 图片中包含二行或者三行的公式
4. 公式类型有两种：赋值和四则运算的公式。两行的包括由一个赋值公式和一个计算公式，三行的包括两个赋值公式和一个计算公式。加号（+） 即使旋转为 x ，仍为加号， * 是乘号
5. 赋值类的公式，变量名为一个汉字。 汉字来自两句诗（不包括逗号）： 君不见，黄河之水天上来，奔流到海不复回 烟锁池塘柳，深圳铁板烧
6. 四则运算的公式包括加法、减法、乘法、分数、括号。 其中的数字为多位数字，汉字为变量，由上面的语句赋值。
7. 输出结果的格式为：图片中的公式，一个英文空格，计算结果。 其中： 不同行公式之间使用英文分号分隔 计算结果时，分数按照浮点数计算，计算结果误差不超过0.01，视为正确。
8. 整个label文件使用UTF8编码

决赛样例：

![](imgs/level2.jpg)

初赛的题不难，只需要识别文本序列即可，决赛的算式比较复杂，需要先经过图像处理，然后才能输入到神经网络中进行端到端的文本序列识别。

### 评价指标

官方的评价指标是准确率，初赛只有整数的加减乘运算，所得的结果一定是整数，所以要求序列与运算结果都正确才会判定为正确。

但决赛的数字通常都是五位数，并且会有很多乘法和加法，以及一定会存在的一个分数，所以结果很容易超出64位浮点数所能表示的范围，因此官方在经过讨论后决定只考虑文本序列的识别，不评价运算结果。

而我们本地除了会使用官方的准确率作为评估标准以外，还会使用 CTC loss 来评估模型。

## 数据的探索

### 定义

决赛的数据集探索就复杂得多，我们先明确两个概念：

`流=42072;圳=86;(圳-(97510*45921))*流/35864`

在这个式子中，`流=42072;圳=86;`被称为赋值式，`(圳-(97510*45921))*流/35864`被称为表达式，赋值式和表达式统称为公式，`+-*/`被称为运算符。

### 分析

首先我们对样本的每个字出现的次数进行了统计：

![](imgs/level2_barplot.png)

可以看到数字的分布很有意思，0出现的次数比其他数字都低，其他的数字出现次数基本一样，所以立即推这是直接按随机数生成的，0不能出现在首位，所以概率变低。

分号和等号出现的次数一样，这是因为每个赋值式都有一个等号和一个分号。它出现的概率是 1.65807，因此可以猜出一个赋值式和两个赋值式的比例是 1:2。

运算符出现的概率都是一样的，所以可以推断它们是直接随机取的。

括号出现的概率是 1.36505，我们统计了一下括号出现的所有可能：

```
1+1+1+1

(1+1)+1+1
1+(1+1)+1
1+1+(1+1)
(1+1+1)+1
1+(1+1+1)

((1+1)+1)+1
(1+(1+1))+1

1+((1+1)+1)
1+(1+(1+1))

(1+1)+(1+1)
```

一共有11种可能，按括号的数量统计括号出现的频率可以得出 2*5/11.0+5/11.0 = 1.3636，因此括号也是从上面几种模板随机取的。

中文除了“不”字出现了两次，概率翻倍，其他字概率基本相等。中文字取自于下面两句诗：“君不见，黄河之水天上来，奔流到海不复回 烟锁池塘柳，深圳铁板烧”，所以也可以推断出是按字直接随机取的。

### 总结

* 中文直接等概率取，“不”概率加倍
* 括号从11种情况中随机取
* 运算符每次必出四个
* 1/3概率取一个赋值式，2/3概率取2个赋值式
* 运算符/永远都会出现一次，中文在上
* 运算符+-*随机取，概率都是1/3
* 数字取值范围是[0, 100000]

## 数据预处理

由于原始的图像十分巨大，直接输入到 CNN 中会有90%以上的区域是没有用的，所以我们需要对图像做预处理，裁剪出有用的部分。然后因为图像有两到三个式子，因此我们采取的方案是从左至右拼接在一起，这样的好处是图像比较小。（900\*80=72000 vs 600\*270=162000）

我主要使用了以下几种技术：

* [转灰度图](http://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
* [直方图均衡](http://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)
* [中值滤波](http://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html)
* [开闭运算](http://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)
* [二值化](http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
* [轮廓查找](http://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html)
* [边界矩形](http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html)

首先先进行初步的关键区域提取：

```py
def plot(index):
    img = cv2.imread('%s/%d.png'%(IMAGE_DIR, index))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eq = cv2.equalizeHist(gray)
    b = cv2.medianBlur(eq, 9)
    
    m, n = img.shape[:2]
    b2 = cv2.resize(b, (n//4, m//4))

    m1 = cv2.morphologyEx(b2, cv2.MORPH_OPEN, np.ones((7, 40)))
    m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((4, 4)))
    _, bw = cv2.threshold(m2, 127, 255, cv2.THRESH_BINARY_INV)
    
    bw = cv2.resize(bw, (n, m))

    r = img.copy()
    img2, ctrs, hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, 
      cv2.CHAIN_APPROX_SIMPLE)
    for ctr in ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(r, (x, y), (x+w, y+h), (0, 255, 0), 10)
```

![](imgs/level2_preprocessing1.png)

### 去噪

首先要将图像[转灰度图](http://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)，然后用初赛使用的[直方图均衡](http://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)提高图像的对比度，这里噪点还在，所以需要进行滤波，我们这里使用了[中值滤波](http://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html)，它能很好地滤掉噪点和干扰线。（上图的 blur）

### 连接公式

现在我们只关心公式的提取，而不在意字符的提取（因为无法保证准确提取），所以我们需要将这些字符连接起来。这里首先对图像进行了4倍的缩放，然后使用了一种叫做[开闭运算](http://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)的算法来连接字符。因为我们要的是横向连接，纵向不需要连接，所以我们选择了 (7, 40) 大小的开运算，然后为了滤掉不必要的噪声，我们使用了 (4, 4) 的闭运算。（位于上图中间的 m2）

### 关键区域提取

在拼接好公式以后，我们就可以对图像使用[轮廓查找](http://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html)的算法了，很容易我们就可以抓到图像的三个边缘点集，然后我们使用[边界矩形](http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html)函数得到矩形的 (x, y, w, h)，完成关键区域提取。提取之后我们将绿色的矩形画在了原图上。（位于上图右下角的 rect）

### 微调

由于之前使用了很大的 kernel 进行滤波，所以这里需要进行一个微调的操作：

```py
# 微调三个公式
d = 20
d2 = 5
imgs = []
sizes = []
for i, ctr in enumerate(ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi = img[max(0, y-d):min(m, y+h+d),max(0, x-d):min(n, x+w+d)]
    p, q, _ = roi.shape
    
    x = b[max(0, y-d):min(m, y+h+d),max(0, x-d):min(n, x+w+d)]
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((3, 3)))
    _, x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY_INV)
    _, x, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(x))
    roi2 = roi[max(0, y-d2):min(p, y+h+d2),max(0, x-d2):min(q, x+w+d2)]
    imgs.append(roi2)
    sizes.append(roi2.shape)
```

首先通过之前的矩形，扩充20像素，然后裁剪出关键区域，这里是直接对滤波的图裁剪，所以分辨率很高。然后经过简单的闭运算滤波，二值化，提取边框，这里即使有噪点也不用担心，裁多了不要紧，裁少了才麻烦，然后裁出来的图可能会比较小，因为滤波过了，所以再扩充5个像素，达到不错的效果。

以下是几个例子：

![](imgs/level2_preprocessing2.png)

![](imgs/level2_preprocessing3.png)

![](imgs/level2_preprocessing4.png)

### 连接三个公式

裁出来准确的公式以后，我们就可以直接进行横向连接了：

```py
# 连接三个公式
sizes = np.array(sizes)
img2 = np.zeros((sizes[:,0].max(), sizes[:,1].sum()+2*(len(sizes)-1), 3),
                dtype=np.uint8)
x = 0
for a in imgs[::-1]:
    w = a.shape[1]
    img2[:a.shape[0], x:x+w] = a
    x += w + 2
```

下图是拼接好的图像：

![](imgs/level2_preprocessing5.png)

### 并行预处理

如果直接使用 python 的 for 循环去跑，只能占用一个核的 CPU 利用率，为了充分利用 CPU，我们使用了多进行并行预处理的方法让每个 CPU 都能满载运行。为了能够实时查看进度，我使用了 tqdm 这个进度条的库。

```py
p = Pool(12)

n = 100000
if __name__ == '__main__':
    rs = []
    for r in tqdm(p.imap_unordered(f, range(n)), total=n):
        rs.append(r)
```

![](imgs/level2_preprocessing6.png)

### 总结

这里我们把各个量之间的关系都画出来了，很有意思。

```py
pd.plotting.scatter_matrix(df, alpha=0.1, figsize=(14,8), diagonal='kde');
```

![](imgs/level2_scatter_matrix.png)

其中的 x, y 表示公式的起始坐标，w, h 表示公式的宽和高，n, m 表示原图的宽和高，r 表示有几个公式。我们可以从图中看到，x, y 没有明显的规律，稍微有一点规律就是越宽的图能得到的 x 越大（废话，宽1000的图不可能有公式出现在1200）。

w 也没有明显的规律，是典型的正态分布，而 h 则有两个峰，这是因为公式有两个和三个的差别。

m, n 很有规律，它们是按某几个固定的数随机取的，m 的取值是从 [400, 500, 600, 700, 800, 900, 1000] 中随机选取的，n 是从 [800, 1600, 2400, 3200, 4000] 中随机取的。

```py
Counter(df['m'])
Counter({400: 14233,
         500: 14414,
         600: 14332,
         700: 14304,
         800: 14293,
         900: 14299,
         1000: 14125})

Counter(df['n'])
Counter({800: 19872, 1600: 19937, 2400: 20128, 3200: 19975, 4000: 20088})
```

## 模型结构

由于我们只对 `base_model` 进行了修改，ctc 部分直接照搬之前的代码即可，因此这里我们只讨论 `base_model`，下面是代码：

```py
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

rnn_size = 128
l2_rate = 1e-5

input_tensor = Input((width, height, 3))
x = input_tensor
for i, n_cnn in enumerate([3, 4, 6]):
    for j in range(n_cnn):
        x = Conv2D(32*2**i, (3, 3), padding='same', kernel_initializer='he_uniform', 
                   kernel_regularizer=l2(l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

# x = AveragePooling2D((1, 2))(x)
cnn_model = Model(input_tensor, x, name='cnn')

input_tensor = Input((width, height, 3))
x = cnn_model(input_tensor)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[3]*conv_shape[2]

print conv_shape, rnn_length, rnn_dimen

x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2
rnn_imp = 0

x = Dense(rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
x = Activation('relu')(x)
# x = Dropout(0.2)(x)

gru_1 = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
gru_1b = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])

# x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
rnn_out = x
base_model = Model(input_tensor, x)
```

在经过多次的代码迭代以后，我将 cnn 打包为了一个 model，这样模型会简洁很多：

![](imgs/level2_basemodel.png)

模型思路是这样的：首先输入一张图，然后通过 cnn 导出 (112, 10, 128) 的特征图，其中112就是输入到 rnn 的序列长度，10 指的是每一条特征的高度是10像素，将后面 (10, 128) 的特征合并成1280，然后经过一个全连接降维到128维，就得到了 (112, 128) 的特征，输入到 RNN 中，然后经过两层双向 GRU 输出112个字的概率，然后用 CTC loss 去优化模型，得到能够准确识别字符序列的模型。

### CNN

CNN 的结构如下图：

![](imgs/level2_cnn_model.png)

理论最大序列长度为46个字符（数字可能为100000，所以是 `2*9+3*6+4+4+2=46`，对于 CTC 来说，我们最好要输入大于最大长度2倍的序列，才能收敛得比较好。之前我直接卷积到50左右了，然后对于连续字符来说，没有空白能将它们分隔开来，所以收敛效果会差很多。这里的最大序列长度我之前总是算错，因为我用的是 Python2，没有 decode 成 utf-8 的话，一个中文占三个字节。

CNN 的结构由原来的两层卷积一层池化，改为了多层卷积，一层池化的结构，由于卷积层分别是3，4和6层，我称之为 346 结构。

### GRU

为什么使用 RNN 呢，这里我举一个很经典的例子：研表究明，汉字的序顺并不定一能影阅响读，比如当你看完这句话后，才发这现里的字全是都乱的。

人眼去阅读一段话的时候，是会顾及到上下文的，不是依次单个字符的识别，因此引入 RNN 去识别上下文能够极大提升模型的准确率。在决赛中，序列有几个地方都是有上下文关系的：

* 前面一个或两个赋值式一定是 中文=数字; 这样的形式
* 左括号一定会有右括号
* 括号的位置是有语法规则的
* 一定会有一个分式
* 分式的分子一定是中文
* 如果只有一个赋值式，那么表达式中的中文一定是赋值式的中文
* 如果有两个赋值式，赋值式容易看清，表达式不容易看清，那么可以通过赋值式的中文去修正表达式的中文，特别是分子中文被裁掉的时候

### 其他参数

相比之前初赛的模型，这里进行了一些修改：

* padding 变为了 same，不然我觉得特征图的高度不够，无法识别分数
* 增加了 l2 正则化，loss loss 变得更大了，但是准确率变得更高了（添加 l2 的部分包括卷积层的 kernel，BN 层的 gamma 和 beta，以及全连接层的 weights 和 bias）
* 各个层的初始化变为了 he_uniform，效果比之前好
* 去掉了 dropout，不清楚影响如何，但是反正有生成器，应该不会出现过拟合的情况
* 修改过 GRU 的 implementation 为2，原因是希望显卡能加速 GRU 的速度，但是似乎速度还不如设置为0，使用 CPU 来跑，所以又改回来了

l2 正则化的参数直接参考了 Xception 论文的 4.3 节给的参数：

> Weight decay: The Inception V3 model uses a weight decay (L2 regularization) rate of 4e-5, which has been carefully tuned for performance on ImageNet. We found this rate to be quite suboptimal for Xception and instead settled for 1e-5.

## 生成器

为了得到更多的数据，提高模型的泛化能力，我使用了一种很简单的数据扩充办法，那就是根据表达式中的中文随机挑选赋值式，组成新的样本。这里我们取了前 350*256=89600 个样本来生成，用之后的 10240 个样本来做验证集，还有一点零头因为太少就没有用了。

导入数据的时候，先读取运算式的图像，然后按中文导入赋值式的图像到字典中。因为字典中的 key 是无序的，所以我们在字典中存的是 list，列表是有序的。

```py
from collections import defaultdict

cn_imgs = defaultdict(list)
cn_labels = defaultdict(list)
ss_imgs = []
ss_labels = []

for i in tqdm(range(n1)):
    ss = df[0][i].decode('utf-8').split(';')
    m = len(ss)-1
    ss_labels.append(ss[-1])
    ss_imgs.append(cv2.imread('crop_split2/%d_%d.png'%(i, 0)).transpose(1, 0, 2))
    for j in range(m):
        cn_labels[ss[j][0]].append(ss[j])
        cn_imgs[ss[j][0]].append(cv2.imread('crop_split2/%d_%d.png'%(i, m-j)).transpose(1, 0, 2))
```

然后实现生成器，这里继承了 keras 里的 Sequence 类：

```py
from keras.utils import Sequence

class SGen(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.X_gen = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        self.y_gen = np.zeros((batch_size, n_len), dtype=np.uint8)
        self.input_length = np.ones(batch_size)*rnn_length
        self.label_length = np.ones(batch_size)*38
    
    def __len__(self):
        return 350*256 // self.batch_size
    
    def __getitem__(self, idx):
        self.X_gen[:] = 0
        for i in range(self.batch_size):
            try:
                random_index = random.randint(0, n1-1)
                cls = []
                ss = ss_labels[random_index]
                cs = re.findall(ur'[\u4e00-\u9fff]', df[0][random_index].decode('utf-8').split(';')[-1])
                random.shuffle(cs)
                x = 0
                for c in cs:
                    random_index2 = random.randint(0, len(cn_labels[c])-1)
                    cls.append(cn_labels[c][random_index2])
                    img = cn_imgs[c][random_index2]
                    w, h, _ = img.shape
                    self.X_gen[i, x:x+w, :h] = img
                    x += w+2
                img = ss_imgs[random_index]
                w, h, _ = img.shape
                self.X_gen[i, x:x+w, :h] = img
                cls.append(ss)

                random_str = u';'.join(cls)
                self.y_gen[i,:len(random_str)] = [characters.find(x) for x in random_str]
                self.y_gen[i,len(random_str):] = n_class-1
                self.label_length[i] = len(random_str)
            except:
                pass
        
        return [self.X_gen, self.y_gen, self.input_length, self.label_length], np.ones(self.batch_size)
```

首先随机取一个表达式，然后用正则表达式找里面的中文，再从{中文：图像数组}的字典中随机取图像，经过之前预处理的方式拼接成一个新的序列。

比如随机取了一个 `85882*(河/76020-37023)-铁`，然后我们从铁的赋值式中随机取一个，再从河的赋值式中随便取一个，拼起来就能得到下图：

![](imgs/level2_generator.png)

可以看到背景颜色是不同的，但是并不影响模型去识别。

## 训练

我们训练的策略是先用 Adam() 默认的学习率 1e-3 快速收敛50代，然后用 Adam(1e-4) 跑50代，达到一个不错的 loss，最后用 Adam(1e-5)微调50代，每一代都保存权值，并且把验证集的准确率跑出来。图中的绿色的线 0.9977 就是按上面的方法训练的模型，

![](imgs/loss.png)

当然我们还尝试过先按 1e-3 的学习率训练20代，然后 1e-4 和 1e-5 交替训练2次，每次训练取验证集 loss 最低的结果继续训练，也就是图中红色的线，虽然速度快，但是准确率不够好。

之后我们将全部训练集都用于训练，得到了蓝色的线，效果和绿色差不多。

## 预测结果

读取测试集的样本，然后用 `base_model` 进行预测，这个过程很简单，就不讲了。

```py
X = np.zeros((n, width, height, channels), dtype=np.uint8)

for i in tqdm(range(n)):
    img = cv2.imread('crop_split2_test/%d.png'%i).transpose(1, 0, 2)
    a, b, _ = img.shape
    X[i, :a, :b] = img

base_model = load_model('model_346_split2_3_%s.h5' % z)
base_model2 = make_parallel(base_model, 4)

y_pred = base_model2.predict(X, batch_size=500, verbose=1)
out = K.get_value(K.ctc_decode(y_pred[:,2:], input_length=np.ones(y_pred.shape[0])*rnn_length)[0][0])[:, :n_len]
```

输出到文件的部分有一点值得一提，就是如何计算出真实值：

```py
ss = map(decode, out)

vals = []
errs = []
errsid = []
for i in tqdm(range(100000)):
    val = ''
    try:
        a = ss[i].split(';')
        s = a[-1]
        for x in a[:-1]:
            x, c = x.split('=')
            s = s.replace(x, c+'.0')
        val = '%.2f' % eval(s)
    except:
#         disp3(i)
        errs.append(ss[i])
        errsid.append(i)
        ss[i] = ''
    
    vals.append(val)
    
with open('result_%s.txt' % z, 'w') as f:
    f.write('\n'.join(map(' '.join, list(zip(ss, vals)))).encode('utf-8'))
    
print len(errs)
print 1-len(errs)/100000.

# output
22
0.99978
```

其中的思路说起来也很简单，就是将表达式中的赋值式中文替换为赋值式的数字，然后直接用 python eval 得到结果，算不出来的直接留空即可。这个0.9977模型的可算率达到了0.99978，也就是说十万个样本里面只有22个样本不可算，当然，实际上还是有一些样本即使可算，也会因为各种原因识别错，比如5和6就是错误的重灾区，某些数字被干扰线切过，导致肉眼都辨认不清等。

## 模型结果融合

模型结果融合的规则很简单，对所有的结果进行次数统计，先去掉空的结果，然后取最高次数的结果即可，其实就是简单的投票。

```py
import glob
import numpy as np
from collections import Counter

def fun(x):
    c = Counter(x)
    c[' '] = 0
    return c.most_common()[0][0]

ss = [open(fname, 'r').read().split('\n') for fname in glob.glob('result_model*.txt')]
s = np.array(ss).T
with open('result.txt', 'w') as f:
    f.write('\n'.join(map(fun, s)))
```

将上面 loss 图中的三个模型结果融合以后，最后得到了0.99868的测试集准确率。

## 其他尝试

### 不定长图像识别

在比赛刚开始的时候，尝试过将图像的宽度设置为 None，也就是不定长的宽度，但是由于无法解决 reshape 的问题，这个方案被否了。

### 分别识别

之前尝试过图像切成几块，分别识别，赋值式和表达式的模型分开，考虑到由于无法得到上下文的信息，可能会丢失一定的准确率，做到一半否掉了这个方案。

### 生成器尝试

我们尝试过写一个生成器，但是由于和官方给的图像差太远，并且实际测试的时候要么是生成的准确率高，官方的准确率低，要么反过来，所以没有投入使用。

![](imgs/level2_generator2.png)

上图第一个是官方的图像，后面五个是我们的生成器生成的，可以看到我们的字没有官方的紧凑，等号也不太一样，分式我们的字又太紧凑了。

### 其他 CNN 模型的尝试

除了自己搭模型，我还尝试过用 ResNet，DenseNet 替换 CNN，然后去训练，但是由于本身这些模型就很大，训练起来速度很慢，然后主要问题又不在模型不够复杂，因为从绘制出来的 loss 曲线来看，虽然前面的 val_loss 一直在抖，但是在第50代学习率下降以后就非常平缓了，这模型是没有过拟合的：

![](imgs/level2_loss.png)

### 替换 GRU 为 LSTM

在比赛最后尝试过将 GRU 替换为 LSTM，得到的结果是十分类似的，但是提交上去以后准确率有轻微下降（多错了几个样本，可能是运气问题），之前做验证码识别的时候也是替换过，效果差不多，因此没有继续尝试。理论上这个序列长度并没有很长，GRU 和 LSTM 影响不大。

![](imgs/level2_loss_lstm.png)

## 总结

### 对项目的思考

本项目中，需要注意以下几个重要的点：

* 数据准备：
 * 深度学习同传统图像处理技术结合，可以达到更好的准确率
 * 文本识别可以构造验证码生成器进行数据增强，增加训练样本数
* 模型优化：
 * 如何根据项目特点，对模型结构进行调整，如CNN 部分减少池化层使用，等等
 * 为了防止过拟合，在模型中引入 L2 正则化
* 模型训练：
 * 使用学习率衰减策略，训练模型
 * 对复杂的模型，可以将同一批次输入数据分摊给多个GPU进行计算。

### 有趣的样本

#### 95170

在测试集里有一个 95170.png 样本很难分割：

![](imgs/level2_95170.png)

因为它的字太浅了，很难被切割出来，肉眼也基本无法分辨。

它的表达式也很难切，稍有不慎就切掉中文了：

![](imgs/level2_95170_0.png)

#### 干扰线

在我们分割的验证集中，发现了被干扰线成功干扰的样本：

![](imgs/level2_sample1.png)

我们可以看到第一个 7 倾斜以后加上一条干扰线，很容易就被模型认成4了，但是人类却不会犯这样的错，这也是 CNN 和 人类之间的区别，目测卷积层自动把图像转灰度图了。

### 可能的改进

* 将真正的生成器写出来，这样就可以获取无穷无尽的样本，而不是使用已有的样本进行拼接，对5和6的识别，以及很多横向的中文的识别，会有很好的帮助，因为它们在已有样本中十分罕见，以至于模型无法准确分辨5和6，以及横向的中文
* 做更好的预处理，比如干扰线和字的颜色是不同的，可以通过程序去除，切图可以更精准一些，可以极大提高训练速度
* 使用其他的模型，比如群里有人提到的 attention 模型，或者看看 OCR 相关的论文，找更多的模型，融合结果，比直接跑类似结构的模型来融合的效果会好很多