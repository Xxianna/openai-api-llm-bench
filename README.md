# openai-api-llm-bench
用于测试openai api的llm的输入输出和并发性能，grok3写的，不太确定是否合理，欢迎任何改进
- 个性化配置，打开py文件
```
client = OpenAI(base_url="http://192.168.31.87:8999/v1", api_key="123")
modelname = "QwQ-32B-FP8-Dynamic"
tokenizer = AutoTokenizer.from_pretrained("./QwQ-32B-AWQ")  #仅用于tokenizer
```
- 使用
```
(C:\PRJ\AI\aienv) PS C:\PRJ\AI> python openaibench.py -t 4 -tg 1000 -pp 4000
Running output test...
[43.094671907648035, 43.59943643819663, 45.395468912461844, 45.344722308508565]
Output test results:
Average token rate: 44.36 ± 1.19 tokens/second

Running input test...
[547.9523854222047, 1811.126067195291, 544.5351086089643, 497.56228167611766]
Input test results:
Average preprocess rate: 850.29 ± 640.97 tokens/second
```
- 说明
```
usage: test.py [-h] [-t T] [-pp PP] [-tg TG]

Test OpenAI API LLM performance

options:
  -h, --help  show this help message and exit
  -t T        Number of concurrent processes
  -pp PP      Number of input tokens for preprocessing test
  -tg TG      Number of output tokens for generation test
```
