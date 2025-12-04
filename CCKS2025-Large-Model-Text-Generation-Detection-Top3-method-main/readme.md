[中文](./readme.md)|[English](./readme_en.md)

比赛链接：https://tianchi.aliyun.com/competition/entrance/532358?spm=a2c22.29524702.0.0.5827526eDLUYLN

## 背景

本文介绍了CCKS2025大模型文本生成检测

> **数据集介绍**
> 本次比赛的数据主要来自通用领域的公开新闻、报道等，所有数据将以JSONL对象的格式进行存储：
> **训练集**
> 训练集涵盖7种大模型：GPT-4o, DeepSeek, Llama3, ChatGPT, GLM-4, Qwen2.5, Claude-3，数据来源涵盖ELI5（问答）、BBC News（新闻写作）、ROC stories（故事生成）、Abstracts（学术写作）、IMDB（评论表达）、Wikipedia（知识解释）共6种任务，训练数据总共包含28000条样本，人类和大模型文本比例为1:1。具体而言，其数据示例如下所示：
> {"text": "Registering a Limited Liability Company (LLC) in a foreign state—meaning a state other than the one where you primarily conduct business—can be a strategic decision, but it involves certain considerations and potential issues:\n\n1. **Foreign Qualification**: If you form an LLC in one state but do business in another, you'll need to register as a foreign LLC in the state where you conduct business. This involves filing additional paperwork and paying fees.\n\n2. **Compliance and Fees**: Foreign qualification typically requires paying registration and ongoing annual fees in both the home state and the foreign state. This can increase your operational costs.\n\n3. **Registered Agent**: You must appoint a registered agent in each state where your LLC is registered. This agent is responsible for receiving official documents and legal papers on behalf of your LLC.\n\n4. **Taxation**: Different states have different tax obligations. Some states may have higher taxes or more complex tax structures, which could affect your business’s bottom line.\n\n5. **Reporting Requirements**: States may have different annual report and renewal requirements. You’ll need to keep track of these to maintain good standing in each state.\n\n6. **Legal Jurisdiction**: Operating in multiple states subjects your LLC to the laws and jurisdiction of those states. This can complicate legal matters if disputes arise.\n\n7. **Operational Complexity**: Managing compliance, taxes, and legal matters in multiple states can increase administrative burdens and complexity.\n\n8. **Business Licenses**: You may need specific licenses or permits to operate legally in a foreign state, depending on your business activities.\n\n9. **Asset Protection and Liability**: Some states offer stronger asset protection laws than others, which might influence your choice. However, operating in multiple states could complicate liability issues.\n\nBefore deciding to register an LLC in a foreign state, it’s advisable to consult with legal and tax professionals who can provide guidance based on your specific business needs and goals.", "label": 1}
> {"text": "Basically there are many categories of \" Best Seller \" . Replace \" Best Seller \" by something like \" Oscars \" and every \" best seller \" book is basically an \" oscar - winning \" book . May not have won the \" Best film \" , but even if you won the best director or best script , you 're still an \" oscar - winning \" film . Same thing for best sellers . Also , IIRC the rankings change every week or something like that . Some you might not be best seller one week , but you may be the next week . I guess even if you do n't stay there for long , you still achieved the status . Hence , # 1 best seller .", "label": 0}
> **测试集**
> 测试集分A榜测试集和B榜测试集，分别包含2800条数据，未知来源，只包含文本内容，不包含标签，其数据样式如下：
> {"text": "Okay! Imagine your mind is like a TV with lots of channels, and each channel is a different thought or feeling. Sometimes, it’s hard to change the channel because there are so many things going on at once.\n\nHypnotism is like having a magical remote control that helps you focus on just one channel at a time. A hypnotist helps you relax and concentrate so you can listen to just one thought. It’s like when someone reads you a bedtime story and you get all cozy and calm.\n\nIn this calm state, you might imagine fun things, like being on a beach or floating on a cloud. It helps you feel relaxed and can sometimes make it easier to learn new things, feel better, or even stop bad habits, like biting your nails.\n\nRemember, it’s all about being super relaxed and using your imagination!" }

## 笔者成绩

A榜单：1/1094

B榜单：3/1094



本文档描述了如何进行方案复现以及训练数据是如何构造的。



# 1 核心逻辑

## 模型

基于Qwen2.5-14B-Instruct进行模型SFT训练，使用LLama Factory框架，Lora方式进行训练。

## 本文档包含

第二节，一个样本的预测流程：这一节介绍了一个样本是如何处理成为训练样本，又是如何进行预测的。

第三节，复现代码介绍。

## 关于训练的几点说明：

### 训练数据构造

- 由于进行数据再平衡的时候（见下文2.6节），存在随机性，训练数据的顺序不完全一致。

- 训练过程中使用的数据，已提供在train_model/train_sft_llm_detect_mask_aug_14B_v8.json里了。

### 模型训练checkpoint选择

- 训练模型使用的是训练step为3000的时候，换算后训练的数据量大于为56000，可以大致理解为训练了2轮。



# 2 一个样本的预测流程

下文将以该文本为示例进行处理，演示整个数据处理流程。

## 2.1 原始文本

以下数据示例为train.jsonl的第5131行，

### 数据示例

```python
json_demo={"text": " But this only works for specific kinds of tumors called \"benign\" tumor cells which grow very slowly into surrounding tissue without invading other parts of the body through blood vessels etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options. But usually when we talk about treating CANCER most times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs pills shots scans surgeries radioactive seeds implants photodynamic agents cryosurgery laser ablation microwave thermotherapy high-intensity focused ultrasound heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet supplements acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies aromatherapy flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning oscillating water column colloidal silver hydrogen peroxide oxygen therapy alkaline mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric chamber colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine B12 injections magnesium sulfate IV vitamin C intravenous injection glucose loading protocols immune modulators antibodies antioxidants herbs adaptogens essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol selenium zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black seed oil garlic ginger bee propolis royal jelly licorice root shiitake mushrooms cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium carbonated spring waters artesian well bottled drinking water distilled purified deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory testing diagnostic procedures medical history physical exams psychological assessment family consultations second opinions follow up visits appointments checkups chronic conditions multiple diseases comorbidities polypharmacy allergies side effects adverse reactions interactions contraindications precautions cautions warnings indications off label use complementary alternative medicine integrative holistic functional non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide", "label": 1}
raw_text = json_demo["text"]
```
## 2.2 计算困惑度标签

### 困惑度（实际计算为交叉熵均值）计算与处理的逻辑如下：

1. 基于Qwen2.5-14B-Instruct计算交叉熵均值，**取每个样本最前面的512个字符计算交叉熵均值，**作为困惑度的代表；
2. 将**训练集**中的样本按照**困惑度的大小进行分组**，每一组给一个标签名称；
3. 训练集外的样本也按照训练集内的困惑度大小进行分组

### 数据示例

该样本的困惑度计算结果为：2.414900541

困惑度（实际计算为交叉熵均值）的标签和大小对应关系如下：

```python
              min       max
ppl_bin                    
非常小      0.159188  1.506109
较小       1.506188  1.732537
小        1.732561  2.077469
中        2.077497  2.426184
大        2.426215  2.816152
较大       2.816399  3.083216
非常大      3.083225  6.387143
```

**上述文本的困惑度标签为：**中



### 关键代码

#### 计算困惑度的函数

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from config import (model2compute_ppl, source_train_json_path, test_set_json_path,
                   train_ppl_path, test_ppl_path)
tqdm.pandas()

# ===========================================================
# 加载模型
# ===========================================================
print("加载模型")
tokenizer = AutoTokenizer.from_pretrained(model2compute_ppl)
model = AutoModelForCausalLM.from_pretrained(
    model2compute_ppl,
    device_map="npu",
    trust_remote_code=True,
    torch_dtype=torch.float16)
# model.to("npu:0")
model = model.eval()  # 切换到评估模式
print("加载模型 done")


# ===========================================================
# 计算困惑度的函数
# ===========================================================
def get_ppl(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 根据模型最大序列长度调整
    )
    input_ids = inputs.input_ids.to("npu")  # 移动到GPU加速

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids  # 自回归任务中，标签与输入相同
        )
        loss = outputs.loss  # 获取损失值
    return loss.item()
```

#### 计算困惑度标签的函数

```python
# 定义分位数边界
quantiles = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
# 计算分位数值
df_train  ## 训练集，其中ppls为计算的交叉熵均值
train_bins = df_train["ppls"].quantile(quantiles).tolist()

def quantile_binning(series):
    # 定义标签
    labels = ['非常小', '较小', '小', '中', '大', '较大', '非常大']
    # 使用pd.cut进行分箱
    binned_series = pd.cut(series, bins=train_bins, labels=labels, include_lowest=True)
    return binned_series
```





## 2.3 文本分割

### 文本分割的逻辑如下：

对于每个文本，先后按照分隔符[[".", "\n",], [":"], [" "]]，进行以下步骤

1. 如果文本的长度小于等于1500，则不进行分割
2. 按照指定分隔符进行分割，比如使用[".", "\n",]则会按照"."与"\n"进行文本分割，得到多个文本块
3. 将分割后的文本块按照文本顺序进行合并，合并后每个文本块之间的重叠的长度为3，合并后的文本长度也保证小于1500

### 数据示例

下文给出上述示例样本(`raw_text`)进行分割后的结果:

```python
raw_text_split=[
' But this only works for specific kinds of tumors called "benign" tumor cells which grow very slowly into surrounding tissue without invading other parts of the body through blood vessels etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options.',

' But usually when we talk about treating CANCER most times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs pills shots scans surgeries radioactive seeds implants photodynamic agents cryosurgery laser ablation microwave thermotherapy high-intensity focused ultrasound heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet supplements acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies aromatherapy flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning oscillating water column colloidal silver hydrogen peroxide oxygen therapy alkaline mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric chamber colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine B12 injections magnesium sulfate IV vitamin C intravenous injection glucose loading protocols immune modulators antibodies antioxidants herbs adaptogens essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol selenium zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black seed oil garlic ginger bee propolis royal jelly licorice root shiitake mushrooms ',

"root shiitake mushrooms cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium carbonated spring waters artesian well bottled drinking water distilled purified deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory testing diagnostic procedures medical history physical exams psychological assessment family consultations second opinions follow up visits appointments checkups chronic conditions multiple diseases comorbidities polypharmacy allergies side effects adverse reactions interactions contraindications precautions cautions warnings indications off label use complementary alternative medicine integrative holistic functional non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide"]
```

### 关键代码

下文中，txt2block为进行文本分割的关键代码

txt2block(input_txt, max_len = 1500, sep=[".", "\n",], interval_row=3)

表示对文本input_txt进行文本分隔，最大长度限制1500个字符，重叠区域为3块，分隔符使用[".", "。"]

```python
def split_txt_equall_length(txts: List[str],
                            max_len: int,
                            interval_row=0,
                            sep="") -> List[str]:
    """用于将文本按照指定长度进行切分，但是不会讲句子从中间切开，不会将完整的句子切开
    """
    ret_txts = []
    # 重新组织数据，将多行数据进行合并，提高判断效率，每段文本最多`max_len`个字符
    new_txt_i = ""
    i = 0
    last_i = 0
    while i < len(txts):
        new_txt_j = txts[i]
        if len(new_txt_j) == 0:
            i += 1
            continue

        if len(new_txt_i) + len(new_txt_j) <= max_len:
            if len(new_txt_i) == 0 or new_txt_i[-1] == sep:
                new_txt_i += new_txt_j
            else:
                new_txt_i += sep + new_txt_j
        else:
            if len(new_txt_i) > 0:
                ret_txts.append(new_txt_i)
                if last_i < i - interval_row:
                    i = max(i - interval_row, 0)
                    new_txt_j = txts[i]
                last_i = i
            new_txt_i = new_txt_j
        i += 1

    if len(new_txt_i) > 0:
        ret_txts.append(new_txt_i)
    return ret_txts

def txt2block(
    input_txt: str, max_len: int, sep=[".", "。"], interval_row=3
) -> List[str]:
    """
    @author: bohanyang
    @将原本固定分隔符的方法修改为了可自定义分隔符

    对文本进行分块，max_len表示每个分块最大的长度。
    对于文本进行如下操作：
    以句号、换行符为分隔符进行分割。

    Args:
        input_txt (str): 输入的文本。
        sep (List[str]): 分隔符列表。
        max_len (int): 每个分块的最大长度。

    Returns:
        List[str]: 分块后的文本列表。
    """
    if len(input_txt) < max_len:
        return [input_txt]
    for separator in sep:
        input_txt = input_txt.replace(separator, separator + "@@<sep>@@")
    ret = [i for i in input_txt.split(r"@@<sep>@@") if len(i) > 0]
    ret = split_txt_equall_length(
        ret,
        max_len,
        interval_row=interval_row,
        sep="",
    )
    return ret
```

## 2.4 语料增强，对语料进行mask

### 语料增强逻辑如下：

1. 每个文本的每个单词都有15%的概率替换为"[MASK]"
2. 一旦某个词被替换为"[MASK]"，后面的五个单词都不再进行mask处理
3. 提示词中进行对应的说明
4. 每个样本，**使用不同随机数种子，**进行**上述步骤5次**

### 数据示例

对上文中raw_text_split中的逐个元素进行mask，其中一个结果如下

```python
"\nBut usually when we talk about treating CANCER [MASK] times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs [MASK] shots scans surgeries radioactive seeds implants [MASK] agents cryosurgery laser ablation microwave thermotherapy high-intensity focused [MASK] heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet [MASK] acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies [MASK] flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning [MASK] water column colloidal silver hydrogen peroxide oxygen therapy [MASK] mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric [MASK] colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine [MASK] injections magnesium sulfate IV vitamin C intravenous [MASK] glucose loading protocols immune modulators antibodies antioxidants herbs [MASK] essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol [MASK] zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black [MASK] oil garlic ginger bee propolis royal jelly licorice root shiitake [MASK]\n"
```

### 关键代码

```python
def mask_random_words(text, alpha, random_state: int=1):
    """
    Randomly masks words in a given text with a 'MASK' token based on a specified probability.

    Parameters:
    - text (str): The input text to process.
    - alpha (float): The probability of masking a word.

    Returns:
    - str: The processed text with some words masked.
    """
    random.seed(random_state)
    words = text.split()
    masked_words = []
    # 控制不要连续多个mask
    skip_time = 0

    for word in words:
        if random.random() < alpha and skip_time == 0:
            masked_words.append("[MASK]")
            skip_time = 5
        else:
            if skip_time > 0:
                skip_time -= 1
            masked_words.append(word)

    return " ".join(masked_words)

# Example usage
text = " Relative calm . Mercs run protection for anyone who can afford it . They are n't as aggressive or salient a presence as coalition forces though . They do n't have a mandate to go kicking doors in when things get heated , they can stay low . The Kurdish north is continuing its slow drift from the rest of Iraq . They still are n't maintaining roads to the south , and everyone else pretty much leaves them alone except for Turkey , who regularly have the odd problem at the border with them ."
alpha = 0.15
masked_text = mask_random_words(text, alpha, random_state=1)
print(masked_text)
# 输出内容如下：
#[MASK] calm . Mercs run protection for anyone [MASK] can afford it . They are n't as aggressive or [MASK] a presence as coalition forces though [MASK] They do n't have a mandate to go [MASK] doors in when things get heated [MASK] they can stay low . The Kurdish north is continuing its slow drift [MASK] the rest of Iraq . They still are n't maintaining roads to the south [MASK] and everyone else pretty much leaves them alone except for Turkey , who regularly have the odd problem at [MASK] border with them .
```

## 2.5 组装提示词

结合2.1~2.5的内容进行提示词构建

提示词模板如下：

```python
# 你提供的模板
instruction =\
"""
- Role: 人工智能文本生成检测专家
- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。
用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），
注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。
- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。
- Workflow:
  1. 接收输入文本,忽略[MASK]的语义信息。
  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。
  3. 根据分析结果，判断文本来源并输出相应的标签。
- Tips:
  1. 分析过程中考虑AI常用的高频词。
  2. 可以结合句子长短，用词风格，语言风格等进行考虑。
  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。
  4. 可以考虑其他这里未提及的重要语义特征。
- user_input:
"""

input_prompt = """
全文困惑度等级为: {}
内容(可能是部分或者全文)：
{}
"""
```

### 数据示例

由于单个样本会进行分片（2.3节文本分割），以及文本增强（2.4节），该样本最终**形成了15个训练样本**。

```json
[    {
        "instruction": "\n- Role: 人工智能文本生成检测专家\n- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。\n用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），\n注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。\n- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。\n- Workflow:\n  1. 接收输入文本,忽略[MASK]的语义信息。\n  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。\n  3. 根据分析结果，判断文本来源并输出相应的标签。\n- Tips:\n  1. 分析过程中考虑AI常用的高频词。\n  2. 可以结合句子长短，用词风格，语言风格等进行考虑。\n  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。\n  4. 可以考虑其他这里未提及的重要语义特征。\n- user_input:\n",
        "input": "\n全文困惑度等级为: 中\n内容(可能是部分或者全文)：\nroot shiitake [MASK] cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium [MASK] spring waters artesian well bottled drinking water distilled [MASK] deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory [MASK] diagnostic procedures medical history physical [MASK] psychological assessment family consultations second opinions follow up visits appointments checkups chronic [MASK] multiple diseases comorbidities polypharmacy allergies [MASK] effects adverse reactions interactions contraindications precautions cautions warnings [MASK] off label use complementary alternative medicine integrative holistic [MASK] non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide\n",
        "id": 21696,
        "output": "<label>1</label>"
    },
    {
        "instruction": "\n- Role: 人工智能文本生成检测专家\n- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。\n用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），\n注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。\n- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。\n- Workflow:\n  1. 接收输入文本,忽略[MASK]的语义信息。\n  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。\n  3. 根据分析结果，判断文本来源并输出相应的标签。\n- Tips:\n  1. 分析过程中考虑AI常用的高频词。\n  2. 可以结合句子长短，用词风格，语言风格等进行考虑。\n  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。\n  4. 可以考虑其他这里未提及的重要语义特征。\n- user_input:\n",
        "input": "\n全文困惑度等级为: 中\n内容(可能是部分或者全文)：\nBut usually when we talk about treating CANCER [MASK] times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs [MASK] shots scans surgeries radioactive seeds implants [MASK] agents cryosurgery laser ablation microwave thermotherapy high-intensity focused [MASK] heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet [MASK] acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies [MASK] flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning [MASK] water column colloidal silver hydrogen peroxide oxygen therapy [MASK] mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric [MASK] colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine [MASK] injections magnesium sulfate IV vitamin C intravenous [MASK] glucose loading protocols immune modulators antibodies antioxidants herbs [MASK] essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol [MASK] zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black [MASK] oil garlic ginger bee propolis royal jelly licorice root shiitake [MASK]\n",
        "id": 21696,
        "output": "<label>1</label>"
    },
...
    {
        "instruction": "\n- Role: 人工智能文本生成检测专家\n- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。\n用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），\n注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。\n- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。\n- Workflow:\n  1. 接收输入文本,忽略[MASK]的语义信息。\n  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。\n  3. 根据分析结果，判断文本来源并输出相应的标签。\n- Tips:\n  1. 分析过程中考虑AI常用的高频词。\n  2. 可以结合句子长短，用词风格，语言风格等进行考虑。\n  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。\n  4. 可以考虑其他这里未提及的重要语义特征。\n- user_input:\n",
        "input": "\n全文困惑度等级为: 中\n内容(可能是部分或者全文)：\nBut this [MASK] works for specific kinds of tumors called \"benign\" tumor cells which grow very slowly into surrounding tissue [MASK] invading other parts of the body through blood [MASK] etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options.\n",
        "id": 21696,
        "output": "<label>1</label>"
    },]
```

## 2.6 模型训练

### 进行不同样本的平衡

使用以下代码，进行样本的再平衡，保证label=0和label=1的样本数量一致。

```python
def balance_label(df, label_col='label'):
    # 计算每个类别的样本数量
    class_counts = df[label_col].value_counts().sort_index()

    # 找出最小的样本数量
    min_count = class_counts.min()

    # 对每个类别进行采样
    balanced_data = []
    for class_label, count in class_counts.items():
        class_data = df[df[label_col] == class_label]
        if count > min_count:
            class_data = class_data.sample(min_count, random_state=42)
        balanced_data.append(class_data)

    # 合并所有采样后的数据
    balanced_df = pd.concat(balanced_data)

    # 重新设置索引
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df = balanced_df.sample(frac=1)

    return balanced_df
```

**模型的训练基于上述流程构成的训练样本，基于Qwen2.5-14B-Instruct，进行SFT训练，取第3000个checkpoint得到。**

## 2.7 训练集外推理——数据构造

进行训练集外推理的时候，数据构造大部分流程和上述步骤一致。

**mask比例调整为：0.04**

**mask使用的随机数进行调整**（备注：此处有运气成分，B榜进行尝试的时候，调整了几次随机数），所使用的随机数种子查看whole_process_b/config.py的random_state_add_seed_out变量。

### 数据示例

```json
    {
        "instruction": "\n- Role: 人工智能文本生成检测专家\n- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。\n用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），\n注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。\n- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。\n- Workflow:\n  1. 接收输入文本,忽略[MASK]的语义信息。\n  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。\n  3. 根据分析结果，判断文本来源并输出相应的标签。\n- Tips:\n  1. 分析过程中考虑AI常用的高频词。\n  2. 可以结合句子长短，用词风格，语言风格等进行考虑。\n  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。\n  4. 可以考虑其他这里未提及的重要语义特征。\n- user_input:\n",
        "input": "\n全文困惑度等级为: 较大\n内容(可能是部分或者全文)：\nEmployers, colleges, community positions these things all require that applicants provide a fullydisclosed criminal record (unless the crime was committed as a minor) if one exists. The stigma of having a criminal record can cost [MASK] individual any of these opportunities when heshe is fully capable of that position. The stigma of being a criminal is the issue here. Once someone is labeled a criminal, it consumes hisher identity.\n",
        "id": 0
    },
    {
        "instruction": "\n- Role: 人工智能文本生成检测专家\n- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。\n用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），\n注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。\n- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。\n- Workflow:\n  1. 接收输入文本,忽略[MASK]的语义信息。\n  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。\n  3. 根据分析结果，判断文本来源并输出相应的标签。\n- Tips:\n  1. 分析过程中考虑AI常用的高频词。\n  2. 可以结合句子长短，用词风格，语言风格等进行考虑。\n  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。\n  4. 可以考虑其他这里未提及的重要语义特征。\n- user_input:\n",
        "input": "\n全文困惑度等级为: 较大\n内容(可能是部分或者全文)：\nThe USChina trade war may begin to affect software developers in China as they question whether access to GitHub will be restricted. GitHub's export control rules state that the company must comply with US export [MASK] which may mean it must follow the same rules that restrict exports to Huawei and other similar companies. The Apache Software Foundation, another opensource distributor in the US, released a statement that said that opensource code was not subject to these trade agreements.\n",
        "id": 1
    },
```

## 2.8 训练集外推理——概率预测

对于一个样本，通过计算该**样本在输出label所对应编号的时候的token的概率归一化结果，作为样本预测概率**。

比如，模型最终输出“\<label\>\0<\\label\>”，计算模型在给出"instruction"+"input"后输出到“\<label\>”的时候，计算输出0和输出1的概率值，对这两个概率值进softmax，作为最终的预测概率值。

### 关键代码

下文中`get_prob`函数的instruction，input_txt对象为需要进行预测的样本的instruction，input_txt字段。

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from config import model_output_dir, model_name, suffixes, out_json_file_path, best_ret_file_path
import re
tqdm.pandas()

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="npu"
)
# Ensure the model is in evaluation mode
model.eval()

# 找到 `1` 和 `0` 的 token ID
token_1_id = tokenizer.convert_tokens_to_ids("1")
token_0_id = tokenizer.convert_tokens_to_ids("0")

def get_prob(instruction, input_txt):
    messages = [
        {"role": "user", "content": instruction+input_txt},
        {"role": "assistant", "content": "<label>"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False
    )
    text = re.search("(.*<label>)<|im_end|>", text, re.S).group(1)
    model_inputs_with_prefix = tokenizer([text], return_tensors="pt").to(model.device)
    

    with torch.no_grad():
        outputs = model(**model_inputs_with_prefix, return_dict=True)
        last_token_logits = outputs.logits[0, -1, :] # Shape (vocab_size,)
        prob_for_1, prob_for_0 = F.softmax(last_token_logits[[token_1_id, token_0_id]], dim=-1)
        
    return prob_for_0.item()
```

## 2.9 预测概率值的集成，概率阈值设定

### 计算逻辑

1. 对于同一个样本，数据集中对应同一个id，由于进行了文本分割以及数据mask，能够构成多个样本；
2. 对同一个id的多个样本计算预测为AI生成概率，得到多个AI生成概率值；
3. 计算多个AI生成概率值的均值mean以及多个概率值的中位数median；
4. 计算mean和median的均值，得到对应于原样本的最终样本由AI生成的概率预测值mean_median;
5. 最终AI生成概率值mean_median大于0.998688126则认为该样本属于AI生成；

### 关于阈值的说明

- 概率阈值0.998688126为A榜中多次测试得到的结果，A榜、B榜均使用该概率阈值




# 3 复现代码

## 3.1 模型训练

### 使用LLama Factory 进行模型训练

文件夹train_model内为训练模型所使用的yaml配置文件、训练数据、conda环境说明。

文件说明：

- 其中environment.yml：conda环境说明

- sft_CCK2025_mask_model-14B-0622-v2.7.yaml：训练时使用的yaml文件，最终模型使用的是checkpoint-3000的模型。
- train_sft_llm_detect_mask_aug_14B_v8.json：为训练所使用的数据

### 使用LLama Factory 进行模型权重融合

为了方便计算token概率，使用LLama Factory进行模型权重合并，同时输出权重。

## 3.2 A榜相关文件

### 复现A榜（代码位于whole_process_a中）
#### step1 修改config文件

注意，其中的model_name要修改为训练后的模型权重，

model2compute_ppl为Qwen2.5-14B-Instruct模型的权重，可以去modelscope自行下载

```python
from pathlib import Path

# mask比例
train_mask_ratio = 0.1
# mask_ratio = 0.04
mask_ratio = 0.04  # todo


# 重复的次数
# random_state_add_seed_out = [168, 158, 1230, 820, 1225, 10086, 10010, 10000]
random_state_add_seed_out = [0, 1, 2, 3, 4]
train_repeat_time = 5
repeat_time = len(random_state_add_seed_out)  # todo

# 用于计算交叉熵损失（用于作为困惑度使用）
model2compute_ppl = "/home/models/Qwen/Qwen2.5-14B-Instruct"

# 训练集路径
# 源文件
source_train_json_path = r"./data/train.jsonl"
# 困惑度计算结果
train_ppl_path = r"./output/ppls_df_train.xlsx"
# 训练集输出json
train_output_json_path = r"./output/train_data.json"

# 测试集路径
# 源文件
test_set_json_path = r"./data/test.jsonl"
# 困惑度计算结果
test_ppl_path = r"./output/ppls_df_data_out_qwen25_13B.xlsx"
# 测试集输出json
test_output_json_path = f"./output/test0521-mask{mask_ratio}-reapt{repeat_time}.json"


# 预测相关的路径
# 预测结果输出路径
model_output_dir = "./model_output"
# model路径
model_name = [修改为模型训练权重文件]
# 样本外数据
out_json_file_path = test_output_json_path
# 输出路径后缀
model_name_tag = Path(model_name).name
out_json_file_path_tag = Path(out_json_file_path).name
suffixes = f"-modelName-{model_name_tag}-mask{mask_ratio}-reapt{repeat_time}-{out_json_file_path_tag}"
# 参考最佳结果
best_ret_file_path = "./a榜提交结果/mean_median-AAA-BEST04-Qwen2.5-14B-lora0622-v2.7-cp3000-mask0.04.txt"
# best_ret_file_path = None

```

#### step2 执行代码

其中，

step1_compute_ppls.py：计算困惑度

step2_get_predict_json_v2.py：构造训练样本及测试集样本

step3_compute_token_prob.py：计算token概率，构造最终预测结果

```shell
python step1_compute_ppls.py && python step2_get_predict_json_v2.py && python step3_compute_token_prob.py
```

### 复现B榜（代码位于whole_process_b中）

#### step1 修改config文件

同上

#### step2 执行代码

同上，其中，

step1_compute_ppls.py：计算困惑度

step2_get_predict_json_v2.py：构造训练样本及测试集样本

step3_compute_token_prob.py：计算token概率，构造最终预测结果

```shell
python step1_compute_ppls.py && python step2_get_predict_json_v2.py && python step3_compute_token_prob.py
```



# 4 未尝试的可用技巧

1. 在分类之前添加思考部分，提高分类准确率（COT构造数据，DAPO、GRPO进行模型训练）
2. 标签修改为带有实际意义的内容
3. 结合大模型水印进行处理，动态
4. agent的方式进行处理
5. RAG动态提示词