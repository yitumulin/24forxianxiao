import json
import re
from typing import Dict, Any, List
from haystack import component

@component
class LLMWorkReplyParser:
    """
    Haystack 组件：解析 LLM 回复为结构化 JSON
    新增功能：根据 Work 节点自动生成 has* 关系
    """

    def __init__(self, enable_repair: bool = True):
        self.enable_repair = enable_repair
        self.trailing_comma_pattern = re.compile(r',\s*([}\]])')
    
    @component.output_types(parsed_reply=Dict[str, Any])
    def run(self, llm_replies: List[str]) -> Dict[str, Any]:
        if not llm_replies:
            return {"parsed_reply": {"nodes": [], "relationships": []}}
        
        llm_reply = llm_replies[0]
        parsed = self.parse(llm_reply)
        
        # 核心修改：从解析结果中提取节点并构建关系
        nodes = parsed.get("nodes", [])
        relationships = self._build_relationships(nodes)
        
        return {"parsed_reply": {
            "nodes": nodes,
            "relationships": relationships
        }}
    
    def parse(self, llm_reply: str) -> Dict[str, Any]:
        cleaned_reply = self._clean_json_formatting(llm_reply)
        
        try:
            return json.loads(cleaned_reply)
        except json.JSONDecodeError:
            print(f"无法解析 LLM 回复: {cleaned_reply}")
            return {"nodes": [], "relationships": []}
    
    def _clean_json_formatting(self, text: str) -> str:
        if text.startswith("data: "):
            text = text[6:].strip()
        text = re.sub(r"<[^>]+>", "", text)
        
        if not (text.startswith("{") and text.endswith("}")):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            text = match.group(0) if match else text
            
        return text.strip()
    
    def _build_relationships(self, nodes: List[Dict]) -> List[Dict]:
        """
        根据节点数据自动构建关系
        规则：从 Work 节点指向其他实体，关系类型为 has+实体类型
        """
        relationships = []
        
        # 查找所有 Work 节点
        work_nodes = [
            node for node in nodes 
            if "Work" in node.get("labels", [])
        ]
        
        # 构建关系
        for work_node in work_nodes:
            work_name = work_node.get("properties", {}).get("name", "")
            
            for other_node in nodes:
                # 跳过 Work 节点自身
                if other_node == work_node:
                    continue
                
                # 获取其他节点的标签（排除 Work）
                other_labels = [
                    label for label in other_node.get("labels", [])
                    if label != "Work"
                ]
                
                if not other_labels:
                    continue
                
                # 使用第一个非 Work 标签作为关系类型
                entity_type = other_labels[0]
                other_name = other_node.get("properties", {}).get("name", "")
                
                # 构建 has* 关系
                relationships.append({
                    "source": work_name,
                    "target": other_name,
                    "type": f"has{entity_type}"
                })
                
        return relationships