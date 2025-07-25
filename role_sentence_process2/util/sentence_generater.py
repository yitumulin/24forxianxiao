from haystack import component
import pprint
import os

# 创建 Sentence Pair Generator Component
@component
class SentencePairGenerator:
    """
    一个 Haystack 组件，用于创建句子三元组并生成中间句子
    """
    
    def __init__(self, generator_function):
        """
        初始化组件
        
        :param generator_function: 生成句子的函数，接受 before_sentence, current_sentence, after_sentence 参数
        """
        self.generator_function = generator_function
        self.output_dir = "output/"
        os.makedirs(self.output_dir, exist_ok=True)
    
    @component.output_types(
        sentence_pairs=list,
        generated_results=list,
        file_path=str
    )
    def run(self, work_name: str, sentences: list):
        """
        主运行方法
        
        :param work_name: 作品名称
        :param sentences: 句子列表
        :return: 包含生成结果的字典
        """
        # 创建句子组合：每个句子与它后第二个句子的组合
        sentence_pairs = []
        for i in range(len(sentences) - 2):
            sentence_pairs.append({
                "before_sentence": sentences[i],
                "current_sentence": sentences[i+1],
                "after_sentence": sentences[i+2],
                "generated_sentence": ""  # 初始化为空，后面会填充
            })
        
        # 生成中间句子并添加到结果中
        generated_results = []
        for idx, pair in enumerate(sentence_pairs):
            # 调用生成函数获取结果
            gen_result = self.generator_function(
                before_sentence=pair["before_sentence"],
                current_sentence=pair["current_sentence"],
                after_sentence=pair["after_sentence"]
            )
            
            # 更新生成句子字段
            sentence_pairs[idx]["generated_sentence"] = gen_result
            generated_results.append(gen_result)
        
        # 准备保存数据
        save_data = {
            "work_name": work_name,
            "total_sentences": len(sentences),
            "generated_pairs": sentence_pairs
        }
        
        # 保存到文件
        file_path = os.path.join(self.output_dir, f"{work_name}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(pprint.pformat(save_data))
        
        # 返回所有结果
        return {
            "sentence_pairs": sentence_pairs,
            "generated_results": generated_results,
            "file_path": file_path
        }


# 使用示例
def generate_intermediate_sentences(before_sentence, current_sentence, after_sentence):
    """示例生成函数 - 实际应用中替换为您的实现"""
    # 这里可以使用您的API调用逻辑
    # 示例实现：组合前后句子生成新句子
    return f"{before_sentence} → {after_sentence} 间的过渡句"

# 创建管道
from haystack import Pipeline

# 1. 创建组件实例
generator_component = SentencePairGenerator(generator_function=generate_intermediate_sentences)

# 2. 创建管道
pipeline = Pipeline()
pipeline.add_component("sentence_pair_generator", generator_component)

# 3. 运行管道
work_name = "朱金彩传"
sentences = [
    {'sentence': '朱金彩传播三九真理', 'subject': '朱金彩'},
    # ... 您提供的所有句子 ...
    {'sentence': '刘婆婆打小姐', 'subject': '刘婆婆'}
]

# 提取原始句子文本
raw_sentences = [item['sentence'] for item in sentences]

# 运行管道
result = pipeline.run({
    "sentence_pair_generator": {
        "work_name": work_name,
        "sentences": raw_sentences
    }
})

# 输出结果
print(f"处理完成: {work_name}")
print(f"生成对数: {len(result['sentence_pair_generator']['sentence_pairs'])}")
print(f"结果文件: {result['sentence_pair_generator']['file_path']}")

# 打印前3个生成结果
print("\n前3个生成结果:")
for i, pair in enumerate(result['sentence_pair_generator']['sentence_pairs'][:3], 1):
    print(f"组合 {i}:")
    print(f"  前句: {pair['before_sentence']}")
    print(f"  当前句: {pair['current_sentence']}")
    print(f"  后句: {pair['after_sentence']}")
    print(f"  生成句: {pair['generated_sentence']}\n")