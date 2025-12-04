[中文](./readme.md)|[English](./readme_en.md)

competition link：https://tianchi.aliyun.com/competition/entrance/532358?spm=a2c22.29524702.0.0.5827526eDLUYLN

# CCKS2025 Large Model Text Generation Detection

This document introduces the solution for the CCKS2025 Large Model Text Generation Detection task.

## Background

> **Dataset Introduction**
> The data for this competition is primarily sourced from public news, reports, and other general-domain texts. All data is stored in JSONL format.
> **Training Set**
> The training set covers 7 large models: GPT-4o, DeepSeek, Llama3, ChatGPT, GLM-4, Qwen2.5, and Claude-3. The data sources span 6 different tasks: ELI5 (Q&A), BBC News (news writing), ROC stories (story generation), Abstracts (academic writing), IMDB (review expression), and Wikipedia (knowledge explanation). The training data contains a total of 28,000 samples, with a 1:1 ratio of human-written to model-generated text. A data example is shown below:
> ```json
> {"text": "Registering a Limited Liability Company (LLC) in a foreign state—meaning a state other than the one where you primarily conduct business—can be a strategic decision, but it involves certain considerations and potential issues:\n\n1. **Foreign Qualification**: If you form an LLC in one state but do business in another, you'll need to register as a foreign LLC in the state where you conduct business. This involves filing additional paperwork and paying fees.\n\n2. **Compliance and Fees**: Foreign qualification typically requires paying registration and ongoing annual fees in both the home state and the foreign state. This can increase your operational costs.\n\n3. **Registered Agent**: You must appoint a registered agent in each state where your LLC is registered. This agent is responsible for receiving official documents and legal papers on behalf of your LLC.\n\n4. **Taxation**: Different states have different tax obligations. Some states may have higher taxes or more complex tax structures, which could affect your business’s bottom line.\n\n5. **Reporting Requirements**: States may have different annual report and renewal requirements. You’ll need to keep track of these to maintain good standing in each state.\n\n6. **Legal Jurisdiction**: Operating in multiple states subjects your LLC to the laws and jurisdiction of those states. This can complicate legal matters if disputes arise.\n\n7. **Operational Complexity**: Managing compliance, taxes, and legal matters in multiple states can increase administrative burdens and complexity.\n\n8. **Business Licenses**: You may need specific licenses or permits to operate legally in a foreign state, depending on your business activities.\n\n9. **Asset Protection and Liability**: Some states offer stronger asset protection laws than others, which might influence your choice. However, operating in multiple states could complicate liability issues.\n\nBefore deciding to register an LLC in a foreign state, it’s advisable to consult with legal and tax professionals who can provide guidance based on your specific business needs and goals.", "label": 1}
> {"text": "Basically there are many categories of \" Best Seller \" . Replace \" Best Seller \" by something like \" Oscars \" and every \" best seller \" book is basically an \" oscar - winning \" book . May not have won the \" Best film \" , but even if you won the best director or best script , you 're still an \" oscar - winning \" film . Same thing for best sellers . Also , IIRC the rankings change every week or something like that . Some you might not be best seller one week , but you may be the next week . I guess even if you do n't stay there for long , you still achieved the status . Hence , # 1 best seller .", "label": 0}
> ```
> **Test Set**
> The test set is divided into Test Set A and Test Set B, each containing 2,800 samples. The source of the text is unknown, and the files only contain the text content without labels. An example is shown below:
> ```json
> {"text": "Okay! Imagine your mind is like a TV with lots of channels, and each channel is a different thought or feeling. Sometimes, it’s hard to change the channel because there are so many things going on at once.\n\nHypnotism is like having a magical remote control that helps you focus on just one channel at a time. A hypnotist helps you relax and concentrate so you can listen to just one thought. It’s like when someone reads you a bedtime story and you get all cozy and calm.\n\nIn this calm state, you might imagine fun things, like being on a beach or floating on a cloud. It helps you feel relaxed and can sometimes make it easier to learn new things, feel better, or even stop bad habits, like biting your nails.\n\nRemember, it’s all about being super relaxed and using your imagination!" }
> ```

## Author's Scores

- **Leaderboard A:** 1 / 1094
- **Leaderboard B:** 3 / 1094

This document describes how to reproduce the solution and how the training data was constructed.



# 1. Core Logic

## Model

The solution is based on Supervised Fine-Tuning (SFT) of the **Qwen2.5-14B-Instruct** model using the **LLaMA Factory** framework with the **LoRA** method.

## What This Document Contains

- **Section 2: Prediction Pipeline for a Single Sample**: This section details how a single raw sample is processed into a training sample and how predictions are made.
- **Section 3: Code for Reproduction**: This section introduces the code needed to reproduce the results.

## Notes on Training

### Training Data Construction

- Due to the randomness involved in data rebalancing (see Section 2.6), the order of the training data may not be exactly the same.
- The data used during training is provided in `train_model/train_sft_llm_detect_mask_aug_14B_v8.json`.

### Model Checkpoint Selection

- The model used was saved at training **step 3000**. This corresponds to training on over 56,000 samples, which can be roughly understood as training for 2 epochs.

# 2. Prediction Pipeline for a Single Sample

The following sections will use a sample from `train.jsonl` (line 5131) to demonstrate the entire data processing workflow.

## 2.1 Original Text

### Data Sample

```python
json_demo={"text": " But this only works for specific kinds of tumors called \"benign\" tumor cells which grow very slowly into surrounding tissue without invading other parts of the body through blood vessels etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options. But usually when we talk about treating CANCER most times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs pills shots scans surgeries radioactive seeds implants photodynamic agents cryosurgery laser ablation microwave thermotherapy high-intensity focused ultrasound heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet supplements acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies aromatherapy flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning oscillating water column colloidal silver hydrogen peroxide oxygen therapy alkaline mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric chamber colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine B12 injections magnesium sulfate IV vitamin C intravenous injection glucose loading protocols immune modulators antibodies antioxidants herbs adaptogens essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol selenium zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black seed oil garlic ginger bee propolis royal jelly licorice root shiitake mushrooms cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium carbonated spring waters artesian well bottled drinking water distilled purified deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory testing diagnostic procedures medical history physical exams psychological assessment family consultations second opinions follow up visits appointments checkups chronic conditions multiple diseases comorbidities polypharmacy allergies side effects adverse reactions interactions contraindications precautions cautions warnings indications off label use complementary alternative medicine integrative holistic functional non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide", "label": 1}
raw_text = json_demo["text"]
```
## 2.2 Calculate Perplexity Label

### Logic for Perplexity Calculation and Labeling

Note: Perplexity is calculated as the mean cross-entropy.

Calculate the mean cross-entropy using Qwen2.5-14B-Instruct. The calculation is performed on the first 512 characters of each sample to represent its perplexity.

Group the samples in the training set based on their perplexity scores and assign a label to each group.

Samples outside the training set (e.g., test set) are grouped using the same perplexity bins defined by the training set.

### Data Sample

The calculated perplexity for this sample is: 2.414900541.

The mapping between perplexity labels and value ranges is as follows:

```python
              min       max
ppl_bin                    
非常小Very Low      0.159188  1.506109
较小 Low       1.506188  1.732537
小 Fairly Low        1.732561  2.077469
中 Medium        2.077497  2.426184
大 Fairly High       2.426215  2.816152
较大 High       2.816399  3.083216
非常大 Very High      3.083225  6.387143
```

Based on this table, the perplexity label for the sample text is: Medium.



### Key Code

#### Function to Calculate Perplexity

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
# Load the model
# ===========================================================
print("Loading model")
tokenizer = AutoTokenizer.from_pretrained(model2compute_ppl)
model = AutoModelForCausalLM.from_pretrained(
    model2compute_ppl,
    device_map="npu",
    trust_remote_code=True,
    torch_dtype=torch.float16)
model = model.eval()  # Switch to evaluation mode
print("Model loaded")

# ===========================================================
# Function to calculate perplexity
# ===========================================================
def get_ppl(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Adjust based on the model's max sequence length
    )
    input_ids = inputs.input_ids.to("npu")  # Move to GPU for acceleration

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids  # In an auto-regressive task, labels are the same as inputs
        )
        loss = outputs.loss  # Get the loss value
    return loss.item()
```

#### Function to Assign Perplexity Labels

```python
# Define quantile boundaries
quantiles = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]

# df_train is the training set DataFrame, where 'ppls' is the calculated mean cross-entropy
# Calculate quantile values
train_bins = df_train["ppls"].quantile(quantiles).tolist()

def quantile_binning(series):
    # Define labels
    labels = ['Very Low', 'Low', 'Fairly Low', 'Medium', 'Fairly High', 'High', 'Very High']
    # Use pd.cut for binning
    binned_series = pd.cut(series, bins=train_bins, labels=labels, include_lowest=True)
    return binned_series
```



## 2.3 Text Splitting

### Logic for Text Splitting

For each text, the following steps are performed sequentially using separators [".", "\n"], then [":"], and finally [" "]:

1. If the text length is less than or equal to 1500 characters, no splitting is performed.

2. Split the text using the specified separators (e.g., . and \n), resulting in multiple text chunks.

3. Merge the split chunks in order, ensuring an overlap of 3 chunks between consecutive merged blocks. The length of each merged block is also kept under 1500 characters.

### Data Sample

The result of splitting the example `raw_text` is as follows:

```python
raw_text_split=[
' But this only works for specific kinds of tumors called "benign" tumor cells which grow very slowly into surrounding tissue without invading other parts of the body through blood vessels etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options.',

' But usually when we talk about treating CANCER most times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs pills shots scans surgeries radioactive seeds implants photodynamic agents cryosurgery laser ablation microwave thermotherapy high-intensity focused ultrasound heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet supplements acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies aromatherapy flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning oscillating water column colloidal silver hydrogen peroxide oxygen therapy alkaline mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric chamber colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine B12 injections magnesium sulfate IV vitamin C intravenous injection glucose loading protocols immune modulators antibodies antioxidants herbs adaptogens essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol selenium zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black seed oil garlic ginger bee propolis royal jelly licorice root shiitake mushrooms ',

"root shiitake mushrooms cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium carbonated spring waters artesian well bottled drinking water distilled purified deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory testing diagnostic procedures medical history physical exams psychological assessment family consultations second opinions follow up visits appointments checkups chronic conditions multiple diseases comorbidities polypharmacy allergies side effects adverse reactions interactions contraindications precautions cautions warnings indications off label use complementary alternative medicine integrative holistic functional non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide"]
```

### Key Code

The key function for text splitting is txt2block.
txt2block(input_txt, max_len = 1500, sep=[".", "\n"], interval_row=3) splits input_txt with a max length of 1500 characters, an overlap of 3 chunks, and uses . and \n as separators.

```python
from typing import List

def split_txt_equally(txts: List[str], max_len: int, interval_row=0, sep="") -> List[str]:
    """
    Splits a list of text strings into chunks of a specified max length
    without breaking sentences in the middle.
    """
    ret_txts = []
    new_txt_i = ""
    i = 0
    last_i = 0
    while i < len(txts):
        new_txt_j = txts[i]
        if len(new_txt_j) == 0:
            i += 1
            continue

        if len(new_txt_i) + len(new_txt_j) <= max_len:
            if len(new_txt_i) == 0 or new_txt_i.endswith(sep):
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

def txt2block(input_txt: str, max_len: int, sep=[".", "."], interval_row=3) -> List[str]:
    """
    Splits text into blocks, where max_len is the maximum length of each block.
    The text is split using separators like periods and newlines.

    Args:
        input_txt (str): The input text.
        sep (List[str]): List of separators.
        max_len (int): Maximum length of each block.
        interval_row (int): Number of chunks to overlap.

    Returns:
        List[str]: A list of text blocks.
    """
    if len(input_txt) < max_len:
        return [input_txt]
    for separator in sep:
        input_txt = input_txt.replace(separator, separator + "@@<sep>@@")
    ret = [i for i in input_txt.split(r"@@<sep>@@") if len(i) > 0]
    ret = split_txt_equally(
        ret,
        max_len,
        interval_row=interval_row,
        sep="",
    )
    return ret
```

## 2.4 Corpus Augmentation via Masking

### Masking Logic

1. Each word in the text has a 15% probability of being replaced with "[MASK]".
2. Once a word is replaced, the next five words are skipped from the masking process.
3. The prompt includes instructions to handle the "[MASK]" token.
4. For each sample, this process is repeated 5 times using different random seeds.

### Data Sample

Here is one possible masked version of a chunk from raw_text_split:

```python
"\nBut usually when we talk about treating CANCER [MASK] times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs [MASK] shots scans surgeries radioactive seeds implants [MASK] agents cryosurgery laser ablation microwave thermotherapy high-intensity focused [MASK] heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet [MASK] acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies [MASK] flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning [MASK] water column colloidal silver hydrogen peroxide oxygen therapy [MASK] mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric [MASK] colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine [MASK] injections magnesium sulfate IV vitamin C intravenous [MASK] glucose loading protocols immune modulators antibodies antioxidants herbs [MASK] essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol [MASK] zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black [MASK] oil garlic ginger bee propolis royal jelly licorice root shiitake [MASK]\n"
```

### Key Code

```python
import random

def mask_random_words(text, alpha, random_state: int = 1):
    """
    Randomly masks words in a given text with a 'MASK' token based on a specified probability.

    Parameters:
    - text (str): The input text to process.
    - alpha (float): The probability of masking a word.
    - random_state (int): Seed for the random number generator.

    Returns:
    - str: The processed text with some words masked.
    """
    random.seed(random_state)
    words = text.split()
    masked_words = []
    # Control to avoid consecutive masks
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
text = "Relative calm . Mercs run protection for anyone who can afford it . They are n't as aggressive or salient a presence as coalition forces though . They do n't have a mandate to go kicking doors in when things get heated , they can stay low . The Kurdish north is continuing its slow drift from the rest of Iraq . They still are n't maintaining roads to the south , and everyone else pretty much leaves them alone except for Turkey , who regularly have the odd problem at the border with them ."
alpha = 0.15
masked_text = mask_random_words(text, alpha, random_state=1)
print(masked_text)
# Output:
# [MASK] calm . Mercs run protection for anyone [MASK] can afford it . They are n't as aggressive or [MASK] a presence as coalition forces though [MASK] They do n't have a mandate to go [MASK] doors in when things get heated [MASK] they can stay low . The Kurdish north is continuing its slow drift [MASK] the rest of Iraq . They still are n't maintaining roads to the south [MASK] and everyone else pretty much leaves them alone except for Turkey , who regularly have the odd problem at [MASK] border with them .
```

## 2.5 Assembling the Prompt

The information from steps 2.1-2.4 is combined to construct the final prompt.

Prompt Template

```python
instruction = """
- Role: AI Text Generation Detection Expert
- Profile: You are an expert in the field of AI text analysis with deep professional knowledge. You have a keen insight into the characteristics of texts generated by large models versus those written by humans, and you can quickly identify the differences between them.
The user will provide content from a text (which could be a partial segment or the full text), along with the perplexity level of the entire text (ranked from low to high: Very Low, Low, Fairly Low, Medium, Fairly High, High, Very High).
Note that "[MASK]" represents a hidden word; you should ignore the semantic information of "[MASK]".
- OutputFormat: If the content is generated by a large model, output: <label>1</label>. If it is written by a human, output: <label>0</label>.
- Workflow:
  1. Receive the input text and ignore the semantic information of [MASK].
  2. Analyze features such as linguistic style, logical coherence, and word usage to determine if the text was generated by a large model.
  3. Based on the analysis, determine the source of the text and output the corresponding label.
- Tips:
  1. Consider high-frequency words commonly used by AI during your analysis.
  2. Take into account sentence length, word choice, and linguistic style.
  3. Pay attention to the style of the language. AI-generated language is often grammatically more perfect, whereas human language can sometimes have unconventional expressions.
  4. Consider other important semantic features not mentioned here.
- user_input:
"""

input_prompt = """
Perplexity level for the full text: {}
Content (may be partial or full):
{}
"""
```

### Data Sample

Because a single original sample is split (Section 2.3) and augmented (Section 2.4), the example text ultimately generates **15 training samples**.

```json
[
    {
        "instruction": "\n- Role: AI Text Generation Detection Expert\n- Profile: You are an expert in the field of AI text analysis... (omitted for brevity)\n- user_input:\n",
        "input": "\nPerplexity level for the full text: Medium\nContent (may be partial or full):\nroot shiitake [MASK] cordyceps sinensis Reishi Maitake lion's mane he shou wu Cordycepin Huperzine A lobster shark cartilage coral calcium [MASK] spring waters artesian well bottled drinking water distilled [MASK] deionized demineralized reverse osmosis ultrafiltered nanofiltration membrane filtration pasteurization irradiation gamma sterilization electron beam radiolysis plasma gas chromatography mass spectrometry nuclear magnetic resonance imaging computer tomography X ray fluoroscopy angiogram mammogram CT scan PET Scan SPECT SCAN MRIs sonograms endoscopy bronchoscopy gastroscopy urology prostate examination gynecology Pap smear cytology pathology laboratory [MASK] diagnostic procedures medical history physical [MASK] psychological assessment family consultations second opinions follow up visits appointments checkups chronic [MASK] multiple diseases comorbidities polypharmacy allergies [MASK] effects adverse reactions interactions contraindications precautions cautions warnings [MASK] off label use complementary alternative medicine integrative holistic [MASK] non invasive minimally invasive conservative aggressive radical conventional traditional interdisciplinary multidisciplinary transcultural international global worldwide\n",
        "id": 21696,
        "output": "<label>1</label>"
    },
    {
        "instruction": "\n- Role: AI Text Generation Detection Expert\n- Profile: You are an expert in the field of AI text analysis... (omitted for brevity)\n- user_input:\n",
        "input": "\nPerplexity level for the full text: Medium\nContent (may be partial or full):\nBut usually when we talk about treating CANCER [MASK] times chemotherapy radiation therapy targeted therapies immunotherapies hormone blockers drugs [MASK] shots scans surgeries radioactive seeds implants [MASK] agents cryosurgery laser ablation microwave thermotherapy high-intensity focused [MASK] heat application gene transfer vaccines CAR T cell transfers stem cell infusions clinical trials experimental drug combinations lifestyle changes diet [MASK] acupuncture massage chiropractic care yoga meditation reiki healing touch energy healings homeopathy herbal remedies [MASK] flower essences gemstones amulets talismans prayer mantra chant recitations sacred geometry crystal grids sound bath music vibration frequency tuning [MASK] water column colloidal silver hydrogen peroxide oxygen therapy [MASK] mineral rich foods probiotics prebiotics detoxification fasting intermittent hypoxia saunas hyperbaric [MASK] colon hydrotherapy coffee enema colonic irrigation chelation Therapy Rife machine [MASK] injections magnesium sulfate IV vitamin C intravenous [MASK] glucose loading protocols immune modulators antibodies antioxidants herbs [MASK] essential oils natural compounds phytochemicals polyphenols flavonoids curcumin resveratrol [MASK] zinc quercetin lycopene beta carotene lutein zeaxanthin coenzyme Q10 alpha lipoic acid NAC glutathione melatonin turmeric black [MASK] oil garlic ginger bee propolis royal jelly licorice root shiitake [MASK]\n",
        "id": 21696,
        "output": "<label>1</label>"
    },
    // ... more samples
    {
        "instruction": "\n- Role: AI Text Generation Detection Expert\n- Profile: You are an expert in the field of AI text analysis... (omitted for brevity)\n- user_input:\n",
        "input": "\nPerplexity level for the full text: Medium\nContent (may be partial or full):\nBut this [MASK] works for specific kinds of tumors called \"benign\" tumor cells which grow very slowly into surrounding tissue [MASK] invading other parts of the body through blood [MASK] etc. If your doctor thinks surgery might work they would need more tests done first such as biopsies before making any decisions about treatment options.\n",
        "id": 21696,
        "output": "<label>1</label>"
    }
]
```

## 2.6 Model Training

### Balancing Samples

The following code is used to rebalance the dataset, ensuring an equal number of samples for label=0 and label=1.

```python
def balance_label(df, label_col='label'):
    # Calculate the number of samples for each class
    class_counts = df[label_col].value_counts().sort_index()

    # Find the minimum sample count
    min_count = class_counts.min()

    # Sample each class
    balanced_data = []
    for class_label, count in class_counts.items():
        class_data = df[df[label_col] == class_label]
        if count > min_count:
            class_data = class_data.sample(min_count, random_state=42)
        balanced_data.append(class_data)

    # Concatenate the balanced data
    balanced_df = pd.concat(balanced_data)

    # Reset index and shuffle
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df = balanced_df.sample(frac=1)

    return balanced_df
```

The model is trained via SFT on the Qwen2.5-14B-Instruct base, using the training samples constructed through the above process. The final model is taken from checkpoint 3000.

## 2.7 Inference on Unseen Data - Data Construction

When preparing unseen data for inference, the process is mostly the same as for training, with two key differences:

- Masking ratio is adjusted to: 0.04

- Different random seeds are used for masking. (Note: This introduced an element of luck. For Leaderboard B, several seeds were tested). The seeds used can be found in the random_state_add_seed_out variable in whole_process_b/config.py.

### Data Sampl

```json
[
    {
        "instruction": "\n- Role: AI Text Generation Detection Expert\n- Profile: You are an expert in the field of AI text analysis... (omitted for brevity)\n- user_input:\n",
        "input": "\nPerplexity level for the full text: High\nContent (may be partial or full):\nEmployers, colleges, community positions these things all require that applicants provide a fullydisclosed criminal record (unless the crime was committed as a minor) if one exists. The stigma of having a criminal record can cost [MASK] individual any of these opportunities when heshe is fully capable of that position. The stigma of being a criminal is the issue here. Once someone is labeled a criminal, it consumes hisher identity.\n",
        "id": 0
    },
    {
        "instruction": "\n- Role: AI Text Generation Detection Expert\n- Profile: You are an expert in the field of AI text analysis... (omitted for brevity)\n- user_input:\n",
        "input": "\nPerplexity level for the full text: High\nContent (may be partial or full):\nThe USChina trade war may begin to affect software developers in China as they question whether access to GitHub will be restricted. GitHub's export control rules state that the company must comply with US export [MASK] which may mean it must follow the same rules that restrict exports to Huawei and other similar companies. The Apache Software Foundation, another opensource distributor in the US, released a statement that said that opensource code was not subject to these trade agreements.\n",
        "id": 1
    }
]
```

## 2.8 Inference on Unseen Data - Probability Prediction

For a given sample, the prediction probability is calculated by normalizing the logits for the label tokens (0 and 1).

For example, if the model is expected to output <label>0</label>, we feed it the instruction + input + <label> and then look at the logits for the next token. We extract the probabilities for the tokens 0 and 1 and apply a softmax function to them to get the final prediction probability.

### Key Code

The instruction and input_txt arguments in the get_prob function correspond to the fields of the sample to be predicted.

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

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="npu"
)
model.eval()

# Find the token IDs for '1' and '0'
token_1_id = tokenizer.convert_tokens_to_ids("1")
token_0_id = tokenizer.convert_tokens_to_ids("0")

def get_prob(instruction, input_txt):
    messages = [
        {"role": "user", "content": instruction + input_txt},
        {"role": "assistant", "content": "<label>"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Extract the text up to the point where the label should be generated
    text = re.search(r"(.*<label>)<|im_end|>", text, re.S).group(1)
    model_inputs_with_prefix = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**model_inputs_with_prefix, return_dict=True)
        last_token_logits = outputs.logits[0, -1, :]  # Shape (vocab_size,)
        # Apply softmax to the logits for tokens '1' and '0'
        prob_for_1, prob_for_0 = F.softmax(last_token_logits[[token_1_id, token_0_id]], dim=-1)
        
    return prob_for_0.item() # Returns probability of being human-written
```

## 2.9 Aggregating Prediction Probabilities and Setting a Threshold

### Calculation Logic

1. A single sample from the original dataset (identified by its id) is expanded into multiple test samples through text splitting and masking.

2. Calculate the "AI-generated" probability for each of these expanded samples.

3. For each original id, calculate the mean (mean) and median (median) of the resulting probability values.

4. The final predicted probability for the original sample is the average of its mean and median (mean_median).

5. A sample is classified as AI-generated if its final mean_median probability is greater than 0.998688126.

### Notes on the Threshold

- The probability threshold 0.998688126 was determined through multiple tests on Leaderboard A and was used for both Leaderboard A and B.




# 3 Code for Reproduction

## 3.1 Model Training

### Using LLaMA Factory for Model Training

The train_model folder contains the YAML configuration file, training data, and Conda environment specification used for training.

File Descriptions:

- environment.yml: Conda environment specification.

- sft_CCK2025_mask_model-14B-0622-v2.7.yaml: The YAML configuration file used for training. The final model is from checkpoint-3000.

- train_sft_llm_detect_mask_aug_14B_v8.json: The data used for training.

### Using LLaMA Factory for Merging Model Weights

To facilitate token probability calculations, LLaMA Factory was used to merge the LoRA weights into the base model and export the final weights.

## 3.2 Leaderboard A Reproduction Files

### Reproducing Leaderboard A (code in whole_process_a)
#### Step 1: Modify the config file

In config.py, ensure that model_name is updated to the path of your trained model weights.
model2compute_ppl should be the path to the Qwen2.5-14B-Instruct model weights, which can be downloaded from ModelScope.

```python
from pathlib import Path

# Masking ratio
train_mask_ratio = 0.1
mask_ratio = 0.04

# Number of repetitions (seeds)
random_state_add_seed_out = [0, 1, 2, 3, 4]
train_repeat_time = 5
repeat_time = len(random_state_add_seed_out)

# Model for calculating cross-entropy loss (as perplexity)
model2compute_ppl = "/home/models/Qwen/Qwen2.5-14B-Instruct"

# Training set paths
source_train_json_path = r"./data/train.jsonl"
train_ppl_path = r"./output/ppls_df_train.xlsx"
train_output_json_path = r"./output/train_data.json"

# Test set paths
test_set_json_path = r"./data/test.jsonl"
test_ppl_path = r"./output/ppls_df_data_out_qwen25_13B.xlsx"
test_output_json_path = f"./output/test0521-mask{mask_ratio}-reapt{repeat_time}.json"

# Prediction-related paths
model_output_dir = "./model_output"
# Path to your trained model weights
model_name = "[UPDATE WITH YOUR TRAINED MODEL PATH]"
out_json_file_path = test_output_json_path

# Output file suffixes
model_name_tag = Path(model_name).name
out_json_file_path_tag = Path(out_json_file_path).name
suffixes = f"-modelName-{model_name_tag}-mask{mask_ratio}-reapt{repeat_time}-{out_json_file_path_tag}"

# Reference best result file
best_ret_file_path = "./a榜提交结果/mean_median-AAA-BEST04-Qwen2.5-14B-lora0622-v2.7-cp3000-mask0.04.txt"
```

#### step2 Execute the scripts

- step1_compute_ppls.py: Calculates perplexity.

- step2_get_predict_json_v2.py: Constructs training and test samples.

- step3_compute_token_prob.py: Calculates token probabilities and generates the final prediction results.

```shell
python step1_compute_ppls.py && python step2_get_predict_json_v2.py && python step3_compute_token_prob.py
```

### Reproducing Leaderboard B (code in whole_process_b)

#### step1 Modify the config file

Follow the same instructions as for Leaderboard A.

#### step2 Execute the scripts

Follow the same instructions as for Leaderboard A.

step1_compute_ppls.py：Calculates perplexity.

step2_get_predict_json_v2.py：Constructs training and test samples.

step3_compute_token_prob.py：Calculates token probabilities and generates the final prediction results.

```shell
python step1_compute_ppls.py && python step2_get_predict_json_v2.py && python step3_compute_token_prob.py
```

# 4 Untried but Potentially Useful Techniques

1. Chain-of-Thought (CoT) Prompting: Add a "thinking" step before classification to improve accuracy (e.g., construct data with CoT and train the model with DAPO/GRPO).
2. Meaningful Labels: Change the labels from 0/1 to more descriptive text.
3. Watermarking: Incorporate detection of large model watermarks.
4. Agent-Based Approach: Use an agent-based framework for the classification task.
5. RAG with Dynamic Prompts: Use Retrieval-Augmented Generation (RAG) to dynamically create more effective prompts.