const translations = {
  "t0": {
    "en": "Skip to content",
    "zh": "跳至正文"
  },
  "t1": {
    "en": "Method",
    "zh": "方法"
  },
  "t2": {
    "en": "Results",
    "zh": "结果"
  },
  "t3": {
    "en": "Resources",
    "zh": "资源"
  },
  "t4": {
    "en": "EMNLP 2026 MAIN · ACCEPTED",
    "zh": "EMNLP 2026 Main · 已接收论文"
  },
  "t5": {
    "en": "Read the graph.<br/><span>One arrow at a time.</span>",
    "zh": "从一支箭头，<br><span>读懂整张流程图。</span>"
  },
  "t6": {
    "en": "Locate an arrowhead. Keep the full image in view. Recover its source, condition, and target — then aggregate the triplets into a directed graph.",
    "zh": "以箭头头部为视觉锚点，保留完整流程图作为上下文，逐条恢复「起点、条件、终点」三元组，再汇聚成可追溯的有向图。"
  },
  "t7": {
    "en": "Read the paper",
    "zh": "阅读论文"
  },
  "t8": {
    "en": "Explore the code",
    "zh": "查看代码"
  },
  "t9": {
    "en": "Poster",
    "zh": "研究海报"
  },
  "t10": {
    "en": "<sup>1</sup> Nanjing University   <sup>2</sup> Lenovo (Beijing) Co., Ltd.",
    "zh": "<sup>1</sup> 南京大学 &nbsp; <sup>2</sup> 联想（北京）有限公司"
  },
  "t11": {
    "en": "INPUT",
    "zh": "输入"
  },
  "t12": {
    "en": "flowchart image",
    "zh": "流程图图像"
  },
  "t13": {
    "en": "OUTPUT",
    "zh": "输出"
  },
  "t14": {
    "en": "(source, condition, target)",
    "zh": "(起点, 条件, 终点)"
  },
  "t15": {
    "en": "Every output points back to an arrow.",
    "zh": "每条输出都能定位到对应箭头。"
  },
  "t16": {
    "en": "evaluation benchmarks",
    "zh": "评测基准"
  },
  "t17": {
    "en": "VLM backbones",
    "zh": "视觉语言模型骨干"
  },
  "t18": {
    "en": "BPMN-VLM F1 vs. fine-tuned E2E",
    "zh": "BPMN-VLM F1 相对微调 E2E 的提升"
  },
  "t19": {
    "en": "FlowLearn downstream QA accuracy",
    "zh": "FlowLearn 下游问答准确率"
  },
  "t20": {
    "en": "Highlighted results use Qwen3-VL-4B. QA uses TextFlow’s released subset; see the evaluation details below.",
    "zh": "以上代表性结果使用 Qwen3-VL-4B；问答在 TextFlow 公开的评测子集上进行，具体设置见下文。"
  },
  "t21": {
    "en": "01 / THE METHOD",
    "zh": "01 / 核心方法"
  },
  "t22": {
    "en": "A local anchor.<br/>A global view.",
    "zh": "局部的视觉锚点，<br>完整的全局上下文。"
  },
  "t23": {
    "en": "Thin connectors and crowded layouts make whole-image extraction difficult. TRACE highlights one arrowhead per copy of the full image, giving the VLM a specific edge to trace while retaining the surrounding context.",
    "zh": "细小连线和密集布局让整图抽取容易遗漏或反转边的方向。TRACE 在每份完整图像中只标记一个箭头头部，让模型集中读取指定连接，同时保留周围信息。"
  },
  "t24": {
    "en": "Synthesize &amp; train",
    "zh": "合成数据与模型训练"
  },
  "t25": {
    "en": "Parse native flowchart sources to obtain aligned arrowhead boxes and image–triplet pairs. Fine-tune the detector and the VLM.",
    "zh": "解析原生流程图文件，自动获得箭头头部框与图像—三元组对，用于微调检测器和视觉语言模型。"
  },
  "t26": {
    "en": "Detect &amp; highlight",
    "zh": "检测与视觉标记"
  },
  "t27": {
    "en": "Locate arrowheads and overlay a highlight marker. Each query keeps the complete flowchart image.",
    "zh": "定位箭头头部并绘制高亮框。每次查询都使用完整流程图，保留全局上下文。"
  },
  "t28": {
    "en": "Trace &amp; aggregate",
    "zh": "追踪与三元组汇聚"
  },
  "t29": {
    "en": "Read source, condition, and target for each marked arrowhead. Combine the outputs into the flowchart graph.",
    "zh": "读取标记箭头对应的起点、条件和终点，将逐条输出汇聚成流程图的结构化表示。"
  },
  "t30": {
    "en": "Offline synthesis and training; online arrowhead detection and triplet extraction. Paper, Figure 1.",
    "zh": "离线阶段合成数据并训练模型；在线阶段检测箭头头部并提取三元组。来源：论文图 1。"
  },
  "t31": {
    "en": "EXPLORE THE QUERY",
    "zh": "交互示例 · 理解查询方式"
  },
  "t32": {
    "en": "Pick an arrowhead.<br/>Follow its triplet.",
    "zh": "选择一个箭头，<br>查看对应三元组。"
  },
  "t33": {
    "en": "This illustrative flowchart shows the query format. Selecting an edge moves the blue highlight; the full diagram stays visible.",
    "zh": "下方示意图展示 TRACE 的查询方式。点击不同的边，蓝色高亮框随之移动，完整流程图始终可见。"
  },
  "t34": {
    "en": "Illustrative example. Outputs are predefined to explain the method.",
    "zh": "此处为方法示意，输出预先设定，不调用在线模型。"
  },
  "t35": {
    "en": "RECOVERED TRIPLET",
    "zh": "提取出的三元组"
  },
  "t36": {
    "en": "Start",
    "zh": "开始"
  },
  "t37": {
    "en": "Read input",
    "zh": "读取输入"
  },
  "t38": {
    "en": "Valid?",
    "zh": "有效？"
  },
  "t39": {
    "en": "Process",
    "zh": "处理"
  },
  "t40": {
    "en": "Reject",
    "zh": "拒绝"
  },
  "t41": {
    "en": "End",
    "zh": "结束"
  },
  "t42": {
    "en": "Yes",
    "zh": "是"
  },
  "t43": {
    "en": "No",
    "zh": "否"
  },
  "t44": {
    "en": "02 / THE EVIDENCE",
    "zh": "02 / 实验结果"
  },
  "t45": {
    "en": "Recovering structure<br/>across visual styles.",
    "zh": "跨越视觉风格，<br>恢复流程图结构。"
  },
  "t46": {
    "en": "Nine benchmarks cover handwritten diagrams, conventional flowcharts, hierarchical BPMN, and generated diagrams of increasing complexity. Compare fine-tuned TRACE and whole-image extraction under the same backbone.",
    "zh": "九个基准覆盖手绘图、常规流程图、具有层级结构的 BPMN，以及不同复杂度的生成图。下表比较相同骨干下微调后的 TRACE 与整图端到端抽取。"
  },
  "t47": {
    "en": "Triplet extraction",
    "zh": "三元组提取"
  },
  "t48": {
    "en": "Benchmark",
    "zh": "评测基准"
  },
  "t49": {
    "en": "End-to-end F1",
    "zh": "端到端 F1"
  },
  "t50": {
    "en": "TRACE F1",
    "zh": "TRACE F1"
  },
  "t51": {
    "en": "Difference",
    "zh": "差值"
  },
  "t52": {
    "en": "Exact F1 (%), paper Table 2. FT = task-specific fine-tuning. These are published results, not measurements from this webpage. <a download=\"\" href=\"assets/results.csv\">Download table</a>",
    "zh": "精确匹配 F1（%），来源：论文表 2。两种方法均经过任务微调。此处展示论文报告结果。<a href='assets/results.csv' download>下载结果表</a>"
  },
  "t53": {
    "en": "The interactive table requires JavaScript. <a href=\"assets/results.csv\">Read the results as CSV</a>, or see Table 2 in the paper.",
    "zh": "交互结果表需要 JavaScript。可<a href='assets/results.csv'>查看 CSV 数据</a>，或阅读论文表 2。"
  },
  "t54": {
    "en": "GENERALIZATION",
    "zh": "跨风格泛化"
  },
  "t55": {
    "en": "Higher F1 on <span class=\"accent\">8 / 9</span><br/>unseen benchmarks.",
    "zh": "在 <span class='accent'>8 / 9</span> 个未见基准上，<br>获得更高的 F1。"
  },
  "t56": {
    "en": "Train on eight benchmarks and evaluate on the held-out one. TRACE improves over E2E on eight; CBD is the exception (83.31 vs. 83.66 F1).",
    "zh": "在八个基准上训练，测试被留出的第九个基准。TRACE 在其中八个上优于 E2E；CBD 是例外（83.31 对 83.66 F1）。"
  },
  "t57": {
    "en": "FlowGen-medium · held out",
    "zh": "FlowGen-medium · 留出测试"
  },
  "t58": {
    "en": "Qwen3-VL-4B, no OCR post-processing. Paper, Appendix B; poster.",
    "zh": "Qwen3-VL-4B，无 OCR 后处理。来源：论文附录 B、研究海报。"
  },
  "t59": {
    "en": "DOWNSTREAM REASONING",
    "zh": "下游推理"
  },
  "t60": {
    "en": "A graph that answers<br/>process questions.",
    "zh": "从结构化图，<br>到流程问答。"
  },
  "t61": {
    "en": "Downstream QA accuracy in percent",
    "zh": "下游问答准确率（%）"
  },
  "t62": {
    "en": "Method",
    "zh": "方法"
  },
  "t63": {
    "en": "TextFlow + GT text",
    "zh": "TextFlow + 真值文本"
  },
  "t64": {
    "en": "TRACE + tool-call",
    "zh": "TRACE + 工具调用"
  },
  "t65": {
    "en": "GT triplets (oracle)",
    "zh": "真值三元组（上界）"
  },
  "t66": {
    "en": "Same Qwen3-VL-4B reasoner; TextFlow receives ground-truth Mermaid. Released subset: 100 FlowLearn and 197 FlowVQA images. Paper, Table 4.",
    "zh": "推理骨干均为 Qwen3-VL-4B；TextFlow 输入真值 Mermaid 文本。公开子集包含 100 张 FlowLearn 和 197 张 FlowVQA 图像。来源：论文表 4。"
  },
  "t67": {
    "en": "03 / BATCHING ARROWHEADS",
    "zh": "03 / 多箭头批量查询"
  },
  "t68": {
    "en": "More arrows per query.<br/>Less time per image.",
    "zh": "一次查询更多箭头，<br>降低单图推理时间。"
  },
  "t69": {
    "en": "Highlight K arrowheads on the same full image and request K numbered triplets. Batching three reduces macro-average latency by 39%, with a 0.49-point F1 decrease relative to K = 1.",
    "zh": "在同一张完整图像中标记 K 个箭头头部，一次请求 K 条编号三元组。K = 3 相比 K = 1 将宏平均延迟降低 39%，F1 仅下降 0.49 个百分点。"
  },
  "t70": {
    "en": "s / image",
    "zh": "秒 / 图像"
  },
  "t71": {
    "en": "relaxed F1 (%)",
    "zh": "宽松匹配 F1（%）"
  },
  "t72": {
    "en": "speed vs. K = 1",
    "zh": "相对 K = 1 的速度"
  },
  "t73": {
    "en": "Macro averages over nine benchmarks, Qwen3-VL-4B on NVIDIA RTX 5880 Ada. E2E: 78.39 F1, 2.73 s/image. Paper, Table 10.",
    "zh": "九个基准的宏平均，Qwen3-VL-4B，NVIDIA RTX 5880 Ada。E2E：78.39 F1、2.73 秒/图像。来源：论文表 10。"
  },
  "t74": {
    "en": "04 / RESEARCH RESOURCES",
    "zh": "04 / 论文与开源资源"
  },
  "t75": {
    "en": "Build on TRACE.",
    "zh": "进一步了解 TRACE。"
  },
  "t76": {
    "en": "Code",
    "zh": "代码"
  },
  "t77": {
    "en": "Data synthesis, detector training, VLM fine-tuning, inference, evaluation, and graph QA.",
    "zh": "数据合成、检测器训练、VLM 微调、推理、评测与基于图的问答代码。"
  },
  "t78": {
    "en": "GitHub repository",
    "zh": "GitHub 仓库"
  },
  "t79": {
    "en": "Paper",
    "zh": "论文"
  },
  "t80": {
    "en": "Method, experimental comparisons, ablations, implementation details, and limitations.",
    "zh": "完整方法、基线比较、消融实验、实现细节与局限性。"
  },
  "t81": {
    "en": "Read PDF",
    "zh": "阅读 PDF"
  },
  "t82": {
    "en": "Poster",
    "zh": "海报"
  },
  "t83": {
    "en": "A visual overview of the framework, cross-style robustness, QA, and multi-arrowhead batching.",
    "zh": "概览方法框架、跨风格泛化、下游问答和多箭头批量查询。"
  },
  "t84": {
    "en": "Read PDF",
    "zh": "阅读 PDF"
  },
  "t85": {
    "en": "Data record",
    "zh": "数据记录"
  },
  "t86": {
    "en": "Synthesized supervision and arrowhead annotations. The current Zenodo record has restricted access.",
    "zh": "合成监督数据与箭头头部标注。当前 Zenodo 记录仍为受限访问。"
  },
  "t87": {
    "en": "View access details",
    "zh": "查看访问说明"
  },
  "t88": {
    "en": "Cite this work",
    "zh": "引用本工作"
  },
  "t89": {
    "en": "Copy BibTeX",
    "zh": "复制 BibTeX"
  },
  "t90": {
    "en": "Accepted to the EMNLP 2026 Main Conference. Provisional paper citation; pages, DOI, and the ACL Anthology URL will be added when the proceedings are available.",
    "zh": "论文已被 EMNLP 2026 Main Conference 接收。当前为临时论文引用；正式论文集发布后将补充页码、DOI 和 ACL Anthology 链接。"
  },
  "t91": {
    "en": "Nanjing University · Lenovo<br/><a href=\"mailto:gcheng@nju.edu.cn\">gcheng@nju.edu.cn</a>",
    "zh": "南京大学 · 联想<br><a href='mailto:gcheng@nju.edu.cn'>gcheng@nju.edu.cn</a>"
  },
  "t92": {
    "en": "Research code: Apache-2.0.<br/>Third-party models and data retain their own terms.",
    "zh": "研究代码采用 Apache-2.0 许可证。<br>第三方模型与数据遵循各自条款。"
  }
};
