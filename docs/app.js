'use strict';
const benchmarkNames = ['FlowVQA', 'CBD', 'FC_A', 'FC_B', 'FlowLearn', 'BPMN-VLM', 'FlowGen-e', 'FlowGen-m', 'FlowGen-h'];
// Exact F1 (%), Table 2 of the supplied paper. Both methods are fine-tuned.
const results = {
  qwen4: {trace:[93.77,82.56,63.91,84.32,93.61,87.07,84.18,85.42,74.82],e2e:[92.28,83.40,64.60,80.42,88.50,64.26,85.87,77.23,68.91]},
  qwen8: {trace:[94.38,85.27,62.36,85.35,94.52,87.08,85.00,84.44,76.21],e2e:[92.57,84.53,63.00,80.59,91.46,71.18,85.31,77.18,70.66]},
  minicpm: {trace:[91.03,74.63,55.81,83.68,80.51,77.58,80.90,69.84,57.60],e2e:[88.44,77.28,39.54,67.97,76.32,61.59,76.41,63.14,52.87]},
  gemma: {trace:[68.88,73.84,41.86,82.20,72.50,62.70,53.50,44.48,33.16],e2e:[80.86,69.12,39.05,69.30,69.92,49.88,46.88,28.72,20.33]},
  llava: {trace:[40.97,63.25,18.32,75.00,82.16,42.27,46.84,30.58,20.10],e2e:[42.67,54.43,12.97,71.46,64.18,18.33,33.85,18.57,11.28]}
};
function updateResults(){const value=results[document.getElementById('backbone').value];const unit=typeof currentLanguage!=='undefined'&&currentLanguage==='zh'?'个百分点':'pp';document.getElementById('results-body').innerHTML=benchmarkNames.map((name,i)=>{const gain=value.trace[i]-value.e2e[i];return `<tr><th scope="row">${name}</th><td>${value.e2e[i].toFixed(2)}</td><td class="trace-score">${value.trace[i].toFixed(2)}</td><td class="${gain>=0?'gain-positive':'gain-negative'}">${gain>=0?'+':'−'}${Math.abs(gain).toFixed(2)} ${unit}</td></tr>`}).join('')}
document.getElementById('backbone').addEventListener('change',updateResults);updateResults();
const edges=[{triplet:'(Start, connectedTo, Read input)',box:[283,78]},{triplet:'(Read input, connectedTo, Valid?)',box:[283,169]},{triplet:'(Valid?, Yes, Process)',box:[123,264]},{triplet:'(Valid?, No, Reject)',box:[443,264]},{triplet:'(Process, connectedTo, End)',box:[235,368]},{triplet:'(Reject, connectedTo, End)',box:[331,368]}];
const edgeControls=document.getElementById('edge-controls');
let selectedEdge=0;
edges.forEach((edge,i)=>{const button=document.createElement('button');button.type='button';button.textContent=`Edge ${i+1}`;button.setAttribute('aria-label',`Highlight edge ${i+1}: ${edge.triplet}`);button.setAttribute('aria-pressed',String(i===0));button.addEventListener('click',()=>selectEdge(i));edgeControls.appendChild(button)});
function selectEdge(i){selectedEdge=i;const rect=document.getElementById('arrow-highlight');rect.setAttribute('x',edges[i].box[0]);rect.setAttribute('y',edges[i].box[1]);document.getElementById('triplet-value').textContent=typeof currentLanguage!=='undefined'&&currentLanguage==='zh'?chineseTriplets[i]:edges[i].triplet;Array.from(edgeControls.children).forEach((button,j)=>button.setAttribute('aria-pressed',String(i===j)))}selectEdge(0);
const batching={1:{seconds:4.00,f1:83.30,rf1:86.52},2:{seconds:2.78,f1:82.92,rf1:86.08},3:{seconds:2.44,f1:82.81,rf1:85.84}};
let selectedK=1;
function updateBatching(k){selectedK=k;const data=batching[k];document.getElementById('k-label').textContent=typeof currentLanguage!=='undefined'?interfaceText[currentLanguage].kLabel(k):`${k} arrowhead${k===1?'':'s'} per query`;document.getElementById('latency-value').textContent=data.seconds.toFixed(2);document.getElementById('latency-bar').style.width=`${data.seconds/4*100}%`;document.getElementById('k-f1').textContent=data.f1.toFixed(2);document.getElementById('k-rf1').textContent=data.rf1.toFixed(2);document.getElementById('k-speed').textContent=`${(4/data.seconds).toFixed(2)}×`;document.querySelectorAll('[data-k]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.k)===k)))}
document.querySelectorAll('[data-k]').forEach(button=>button.addEventListener('click',()=>updateBatching(Number(button.dataset.k))));
document.getElementById('copy-citation').addEventListener('click',async()=>{const status=document.getElementById('copy-status'),text=interfaceText[currentLanguage];try{await navigator.clipboard.writeText(document.getElementById('citation-text').textContent);status.textContent=text.copied}catch(error){const selection=window.getSelection(),range=document.createRange();range.selectNodeContents(document.getElementById('citation-text'));selection.removeAllRanges();selection.addRange(range);status.textContent=text.copyFallback}});
