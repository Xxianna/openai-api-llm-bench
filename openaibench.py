import argparse
import time
import statistics
from multiprocessing import Pool
import os
from openai import OpenAI
import tiktoken
import random
from transformers import AutoTokenizer

random.seed(time.time())

client = OpenAI(base_url="http://192.168.31.87:8999/v1", api_key="123")
modelname = "QwQ-32B-FP8-Dynamic"
tokenizer = AutoTokenizer.from_pretrained("./QwQ-32B-AWQ")

# 使用 cl100k_base 编码器，适用于 gpt-3.5-turbo
# tokenizer = tiktoken.encoding_for_model("QwQ-32B-AWQ")


def generate_random_token_string(target_tokens):
    """
    使用 QwQ-32B-AWQ tokenizer 的词汇表生成定长的随机 token 字符串。
    
    Args:
        target_tokens (int): 目标 token 数
        
    Returns:
        str: token 数等于 target_tokens 的随机字符串
    """
    # 获取词汇表
    vocab = list(tokenizer.get_vocab().keys())
    # 过滤掉特殊 token（如 <pad>, <s> 等）和非单词字符
    word_pool = [token for token in vocab if token.isalnum() and not token.startswith("<")]
    
    if not word_pool:
        raise ValueError("No valid tokens found in the vocabulary.")
    
    result = []
    current_tokens = 0
    
    while current_tokens < target_tokens:
        # 随机选择一个 token
        token = random.choice(word_pool)
        # 计算添加该 token 后的 token 数
        temp_text = " ".join(result + [token])
        token_count = len(tokenizer.encode(temp_text, add_special_tokens=False))
        
        if token_count <= target_tokens:
            result.append(token)
            current_tokens = token_count
        else:
            break
    
    # 如果 token 数不够，补齐空格
    while current_tokens < target_tokens:
        result.append(" ")
        current_tokens = len(tokenizer.encode(" ".join(result), add_special_tokens=False))
    
    return " ".join(result)

def output_test(target_tokens):
    """输出测试：输入一个字符，生成目标 token 数，计算每秒输出 token 数"""
    # prompt = "a"
    prompt = generate_random_token_string(5)
    start_time = time.time()
    accumulated_text = ""
    
    # 使用流式生成
    stream = client.chat.completions.create(
        model=modelname,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target_tokens,
        stream=False
    )
    response_content = stream.choices[0].message.content
    token_count = len(tokenizer.encode(response_content, add_special_tokens=False))
    
    # 逐块累积输出，直到达到目标 token 数
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         accumulated_text += chunk.choices[0].delta.content
    #         token_count = len(tokenizer.encode(accumulated_text))
    #         if token_count >= target_tokens:
    #             break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    token_rate = token_count / elapsed_time  # 每秒输出 token 数
    return token_rate

def input_test(prompt_tokens):
    """输入测试：输入指定 token 数，生成 1 个 token，计算每秒预处理 token 数"""
    # 生成指定 token 数的提示，每个 "a" 为 1 个 token
    # prompt = "a" * prompt_tokens
    prompt = generate_random_token_string(prompt_tokens)
    start_time = time.time()
    
    # 使用流式生成，只生成 1 个 token
    stream = client.chat.completions.create(
        model=modelname,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        stream=False
    )
    
    # 测量首个 token 回复的延迟
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         end_time = time.time()
    #         break
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    preprocess_rate = prompt_tokens / elapsed_time  # 每秒预处理 token 数
    return preprocess_rate

def run_test(test_func, num_processes, param):
    """运行指定测试，使用多进程并发"""
    with Pool(processes=num_processes) as pool:
        results = pool.map(test_func, [param] * num_processes)
    return results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test OpenAI API LLM performance")
    parser.add_argument("-t", type=int, default=1, help="Number of concurrent processes")
    parser.add_argument("-pp", type=int, help="Number of input tokens for preprocessing test")
    parser.add_argument("-tg", type=int, help="Number of output tokens for generation test")
    args = parser.parse_args()

    # 如果提供了 -tg，执行输出测试
    if args.tg:
        print("Running output test...")
        results = run_test(output_test, args.t, args.tg)
        avg_rate = statistics.mean(results)
        std_dev = statistics.stdev(results) if len(results) > 1 else 0
        print(results)
        print(f"Output test results:")
        print(f"Average token rate: {avg_rate:.2f} ± {std_dev:.2f} tokens/second")
        # print(f"Standard deviation: {std_dev:.2f} tokens/second")

    # 如果提供了 -pp，执行输入测试
    if args.pp:
        print("\nRunning input test...")
        results = run_test(input_test, args.t, args.pp)
        print(results)
        avg_rate = statistics.mean(results)
        std_dev = statistics.stdev(results) if len(results) > 1 else 0
        print(f"Input test results:")
        print(f"Average preprocess rate: {avg_rate:.2f} ± {std_dev:.2f} tokens/second")
        # print(f"Standard deviation: {std_dev:.2f} tokens/second")

if __name__ == "__main__":
    main()
