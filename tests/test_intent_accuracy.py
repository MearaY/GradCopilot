"""
M-02 IntentClassifier 真实 LLM 准确率评测。

本测试文件使用真实 LLM API 调用（非 Mock），测量当前 prompt 对各意图的
分类准确率。运行前需确保 .env 配置正确、服务可达。

运行方式（在项目根目录）：
    pytest tests/test_intent_accuracy.py -v -s

评测指标：
    - 整体准确率 >= 80% 才视为 prompt 合格（可上线）
    - 任意单个 intent 类别准确率 < 60% 须优化 prompt
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# 加载 .env（真实 API Key）
load_dotenv(Path(__file__).parent.parent / ".env")

# 确保路径正确
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# 测试数据集
# 格式: (用户输入, 期望 intent)
# 覆盖五个意图，每个 10 条，共 50 条
# 意图枚举：rag_query / paper_search / paper_download / build_knowledge / general_chat
INTENT_TEST_CASES = [

    # ── rag_query（10条）
    # 用户在询问技术内容、论文方法、实验结果、概念解释
    ("transformer 的核心创新是什么？", "rag_query"),
    ("Dilated Neighborhood Attention 是什么原理？", "rag_query"),
    ("这篇论文的实验结果达到了什么指标？", "rag_query"),
    ("注意力机制中 Query、Key、Value 分别是什么作用？", "rag_query"),
    ("DiNAT 相比 Swin Transformer 有什么改进？", "rag_query"),
    ("论文中提到的 panoptic segmentation 方法是怎么实现的？", "rag_query"),
    ("介绍一下这篇文章的 related work 部分", "rag_query"),
    ("DINO 和 DETR 有什么关系？", "rag_query"),
    ("这篇论文的 ablation study 说明了什么？", "rag_query"),
    ("局部注意力和全局注意力的区别是什么？", "rag_query"),

    # ── paper_search（10条）
    # 用户想在 arXiv 上搜索、查找、寻找论文
    ("帮我搜索 vision transformer 的最新论文", "paper_search"),
    ("在 arXiv 上找一下 diffusion model 相关的文章", "paper_search"),
    ("我想找一些关于 3D object detection 的论文", "paper_search"),
    ("search for papers about large language models", "paper_search"),
    ("帮我查一下 NeRF 方向最近有什么新工作", "paper_search"),
    ("找找 CVPR 2023 关于目标检测的论文", "paper_search"),
    ("搜一下 point cloud processing 的最新进展", "paper_search"),
    ("有没有关于 contrastive learning 的综述论文", "paper_search"),
    ("我需要找 semantic segmentation 相关的 SOTA 方法", "paper_search"),
    ("帮我找 GPT-4 的技术报告", "paper_search"),

    # ── paper_download（10条）
    # 用户想下载特定论文
    ("下载这篇论文", "paper_download"),
    ("把刚才搜到的第 2 篇下载下来", "paper_download"),
    ("帮我下载 Attention is All You Need 这篇论文", "paper_download"),
    ("download paper 2312.00001", "paper_download"),
    ("把前三篇都下载", "paper_download"),
    ("将搜索结果的第 1 和第 3 篇下载到本地", "paper_download"),
    ("下载 ViT 的论文", "paper_download"),
    ("我要下载这几篇文章", "paper_download"),
    ("获取这篇论文的 PDF", "paper_download"),
    ("帮我保存刚才那篇文章", "paper_download"),

    # ── build_knowledge（10条）
    # 用户想构建/更新/索引知识库
    ("帮我构建知识库", "build_knowledge"),
    ("把下载的论文建立向量索引", "build_knowledge"),
    ("更新一下知识库", "build_knowledge"),
    ("build the knowledge base", "build_knowledge"),
    ("对已下载的文章做向量化处理", "build_knowledge"),
    ("重新索引一下这些论文", "build_knowledge"),
    ("把这些 PDF 加入到知识库里", "build_knowledge"),
    ("我想让你能够检索这些文章，帮我建索引", "build_knowledge"),
    ("知识库需要更新了，帮我重建", "build_knowledge"),
    ("把本次下载的论文入库", "build_knowledge"),

    # ── general_chat（10条）
    # 纯对话，不需要工具或知识库
    ("你好，介绍一下你自己", "general_chat"),
    ("今天天气怎么样", "general_chat"),
    ("帮我用中文写一封邮件", "general_chat"),
    ("什么是机器学习？用简单语言解释", "general_chat"),
    ("Python 和 Java 哪个更适合初学者？", "general_chat"),
    ("帮我写一段 PyTorch 代码", "general_chat"),
    ("深度学习和机器学习有什么区别？", "general_chat"),
    ("你能做什么？", "general_chat"),
    ("谢谢你的帮助", "general_chat"),
    ("怎么提升论文写作能力？", "general_chat"),
]

# 评测准入线：整体 >= 80%，单类别 >= 60%
OVERALL_PASS_THRESHOLD = 0.80    # 整体准确率 >= 80%
PER_CLASS_PASS_THRESHOLD = 0.60  # 单类准确率 >= 60%


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def mock_redis_and_pg(mocker_module_scope=None):
    """
    不 mock LLM，但 mock 掉 Redis（get_history）避免依赖真实 Redis。
    """
    pass  # 在每个测试中单独处理


# ── 辅助函数 ──────────────────────────────────────────────────────

def _run_classification(session_id: str, user_input: str) -> dict:
    """调用真实 classify_intent（不 mock LLM）。"""
    from unittest.mock import patch
    # 只 mock Redis 历史，LLM 保持真实
    with patch("src.modules.intent_classifier.get_history", return_value=[]):
        from src.modules.intent_classifier import classify_intent
        return classify_intent(session_id, user_input)


# ── 核心准确率测试 ─────────────────────────────────────────────────

class TestIntentAccuracy:
    """
    真实 LLM 意图分类准确率评测。
    """

    @pytest.fixture(autouse=True)
    def _load_env(self):
        """确保环境变量已加载。"""
        assert os.environ.get("API_KEY"), "API_KEY 未配置，无法进行真实 LLM 评测"
        assert os.environ.get("BASE_URL"), "BASE_URL 未配置"

    def test_overall_accuracy(self):
        """
        对全部 50 条测试用例运行真实 LLM，整体准确率必须 >= 80%。
        """
        correct = 0
        total = len(INTENT_TEST_CASES)
        failures = []

        # 保存所有结果供后续用例复用，避免重复发 50 次 API
        global _CACHED_RESULTS
        _CACHED_RESULTS = []

        print()  # 换行，开始打印进度
        for i, (user_input, expected_intent) in enumerate(INTENT_TEST_CASES):
            print(f"  [{i+1:02d}/{total}] 测试输入: {user_input[:25]:<25} ", end="", flush=True)
            
            result = _run_classification(f"acc_test_{i}", user_input)
            actual_intent = result["intent"]
            confidence = result["confidence"]
            
            _CACHED_RESULTS.append(result)

            if actual_intent == expected_intent:
                correct += 1
                print(f"-> ✓ {actual_intent} (conf: {confidence:.2f})")
            else:
                failures.append({
                    "input": user_input,
                    "expected": expected_intent,
                    "actual": actual_intent,
                    "confidence": confidence,
                    "reasoning": result.get("reasoning", ""),
                })
                print(f"-> ✗ {actual_intent} (预期: {expected_intent})")

        accuracy = correct / total

        # 打印详细报告（-s 模式可见）
        print(f"\n{'='*60}")
        print(f"  Intent 分类准确率评测报告")
        print(f"{'='*60}")
        print(f"  总样本数: {total}")
        print(f"  正确数:   {correct}")
        print(f"  整体准确率: {accuracy:.1%}  （合格线: {OVERALL_PASS_THRESHOLD:.0%}）")

        if failures:
            print(f"\n  ── 错误样本 ({len(failures)} 条) ──")
            for f in failures:
                print(f"  输入: {f['input']!r}")
                print(f"    期望: {f['expected']}  实际: {f['actual']}  置信度: {f['confidence']:.2f}")
                print(f"    推理: {f['reasoning']}")
                print()
        print(f"{'='*60}\n")

        assert accuracy >= OVERALL_PASS_THRESHOLD, (
            f"整体准确率 {accuracy:.1%} 低于合格线 {OVERALL_PASS_THRESHOLD:.0%}，"
            f"需优化 intent_classify prompt。\n"
            f"失败样本：{failures}"
        )

    def test_per_class_accuracy(self):
        """
        每个 intent 类别单独统计准确率，任意类别不得低于 60%。
        """
        by_class: dict[str, list[bool]] = defaultdict(list)

        # 直接使用上一用例缓存的结果，避免重复发网络请求
        global _CACHED_RESULTS
        if not globals().get("_CACHED_RESULTS"):
            pytest.skip("依赖 overall_accuracy 执行")

        for i, (user_input, expected_intent) in enumerate(INTENT_TEST_CASES):
            result = _CACHED_RESULTS[i]
            actual_intent = result["intent"]
            by_class[expected_intent].append(actual_intent == expected_intent)

        print(f"\n{'='*60}")
        print(f"  各类别准确率")
        print(f"{'='*60}")

        failed_classes = []
        for intent_class, results in sorted(by_class.items()):
            acc = sum(results) / len(results)
            status = "✓" if acc >= PER_CLASS_PASS_THRESHOLD else "✗"
            print(f"  {status} {intent_class:<20}  {acc:.1%}  ({sum(results)}/{len(results)})")
            if acc < PER_CLASS_PASS_THRESHOLD:
                failed_classes.append((intent_class, acc))

        print(f"{'='*60}\n")

        assert not failed_classes, (
            f"以下类别准确率低于合格线 {PER_CLASS_PASS_THRESHOLD:.0%}：\n"
            + "\n".join(f"  {cls}: {acc:.1%}" for cls, acc in failed_classes)
        )

    @pytest.mark.parametrize("user_input,expected", [
        # 边界用例：容易混淆的输入
        ("帮我检索关于 DETR 的内容", "rag_query"),      # 检索≠搜索
        ("在 arXiv 上搜 DETR", "paper_search"),         # 明确提到 arXiv
        ("下载第一篇", "paper_download"),                # 隐式下载
        ("建一下知识库", "build_knowledge"),             # 口语化
        ("用已有论文回答我的问题", "rag_query"),         # 明确引用已有
    ])
    def test_boundary_cases(self, user_input, expected):
        """
        边界测试：容易混淆的意图，LLM 必须正确区分。
        [REF: product.md#F-02]
        """
        result = _run_classification("boundary_test", user_input)
        actual = result["intent"]
        confidence = result["confidence"]

        print(f"\n  边界用例: {user_input!r}")
        print(f"  期望: {expected}  实际: {actual}  置信度: {confidence:.2f}")
        print(f"  推理: {result.get('reasoning', '')}")

        assert actual == expected, (
            f"边界用例分类错误：{user_input!r}\n"
            f"期望 {expected}，实际 {actual}（confidence={confidence:.2f}）"
        )


# ── 结构性测试（不依赖真实 LLM）──────────────────────────────────

def test_test_dataset_coverage():
    """
    验证测试数据集结构完整性：每个 intent 类别必须有 >= 5 条样本。
    [REF: product.md#F-02 意图枚举]
    """
    from src.prompts.intent_classify import VALID_INTENTS
    counts = defaultdict(int)
    for _, intent in INTENT_TEST_CASES:
        counts[intent] += 1

    for intent in VALID_INTENTS:
        assert counts[intent] >= 5, (
            f"意图 '{intent}' 在测试集中只有 {counts[intent]} 条，至少需要 5 条"
        )
