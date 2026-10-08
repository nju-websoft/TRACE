'use strict';
var currentLanguage = 'zh';
const interfaceText = {
  zh:{edge:'边',highlight:'高亮边',backbone:'模型骨干',pp:'个百分点',kLabel:k=>`每次查询 ${k} 个箭头头部`,copied:'引用已复制。',copyFallback:'引用已选中，请按 Ctrl+C 或 Cmd+C 复制。'},
  en:{edge:'Edge',highlight:'Highlight edge',backbone:'Backbone',pp:'pp',kLabel:k=>`${k} arrowhead${k===1?'':'s'} per query`,copied:'Citation copied.',copyFallback:'Citation selected. Press Ctrl+C or Cmd+C to copy.'}
};
const chineseTriplets=['(开始, connectedTo, 读取输入)','(读取输入, connectedTo, 有效？)','(有效？, 是, 处理)','(有效？, 否, 拒绝)','(处理, connectedTo, 结束)','(拒绝, connectedTo, 结束)'];
function setLanguage(language){
  currentLanguage=language==='en'?'en':'zh';
  document.documentElement.lang=currentLanguage==='zh'?'zh-CN':'en';
  document.querySelectorAll('[data-i18n]').forEach(element=>{const value=translations[element.dataset.i18n][currentLanguage];element.innerHTML=value});
  const text=interfaceText[currentLanguage];
  document.getElementById('backbone-label').textContent=text.backbone;
  document.querySelectorAll('[data-language]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.language===currentLanguage)));
  document.getElementById('edge-controls').setAttribute('aria-label',currentLanguage==='zh'?'选择流程图中的边':'Select a flowchart edge');
  document.querySelector('.demo-visual svg').setAttribute('aria-label',currentLanguage==='zh'?'示意流程图：开始、读取输入、有效性判断、处理或拒绝、结束。':'Illustrative flowchart: Start, Read input, Valid decision, Process or Reject, and End.');
  document.querySelector('.k-controls').setAttribute('aria-label',currentLanguage==='zh'?'每次查询的箭头头部数量':'Number of arrowheads per query');
  Array.from(edgeControls.children).forEach((button,i)=>{button.textContent=`${text.edge} ${i+1}`;button.setAttribute('aria-label',`${text.highlight} ${i+1}`)});
  updateResults();selectEdge(selectedEdge);updateBatching(selectedK);
  document.getElementById('copy-status').textContent='';
  const title=currentLanguage==='zh'?'TRACE — 以箭头为中心的流程图理解':'TRACE — Arrowhead-Centric Flowchart Understanding';
  const description=currentLanguage==='zh'?'TRACE 以箭头头部为锚点，逐条恢复流程图三元组。了解核心方法、九个基准上的实验结果，以及论文和开源代码。':'TRACE recovers flowchart graphs one arrowhead at a time. Explore the method, results on nine benchmarks, and research code.';
  document.title=title;document.querySelector('meta[name="description"]').content=description;document.querySelector('meta[property="og:title"]').content=title;document.querySelector('meta[property="og:description"]').content=description;
}
document.querySelectorAll('[data-language]').forEach(button=>button.addEventListener('click',()=>{setLanguage(button.dataset.language);const url=new URL(window.location.href);if(currentLanguage==='en')url.searchParams.set('lang','en');else url.searchParams.delete('lang');history.replaceState(null,'',url)}));
setLanguage(new URLSearchParams(window.location.search).get('lang')==='en'?'en':'zh');
