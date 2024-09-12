# huggingface-test
huggingface-test
```
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download --resume-download Qwen2-0.5B-Instruct --local-dir Qwen2-0.5B-Instruct 
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

下载模型文件
```
git clone https://hf-mirror.com/Qwen/Qwen2-0.5B-Instruct  
```
安装python包依赖，这三个python包都比较大，下载时间较长。  
```
pip install transformers torch accelerate  
```
国内可以指定pip安装源。
```
pip install transformers torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

拷贝官方提供的快速启动脚本，这里我修改模型的路径为绝对路径"/root/Qwen2-0.5B-Instruct"，Prompt修改为"你是谁？"，增加了打印响应结果。

```
vim /root/qwen0.5b-demo.py

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/root/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

prompt = "你是谁？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

Qwen Gradio 图形化界面

```
pip install gradio
```

运行脚本，-- server-name 0.0.0.0允许所有地址进行访问，--checkpoint-path /root/Qwen2-0.5B-Instruct指定模型文件所在目录。  
```
py web_demo.py --server-name 0.0.0.0 --checkpoint-path Qwen/Qwen2-0.5B-Instruct

ruslanmv/Medical-Llama3-8B

```
