/*!
 * 護眼燈市場調查 — 廣告 AB 測試洞察報告 · 應用腳本
 * ---------------------------------------------------------------
 * 內容：來源分析 Tab 切換、Chart.js 圖表初始化
 * 依賴：Chart.js v4.4.1（assets/js/chart.umd.min.js，MIT License）
 * ---------------------------------------------------------------
 */

/* ----- Tab 切換（各來源分析區塊） ----- */
function switchSourceTab(id){
  var ids = ['ptt','yt','ec'];
  for (var i = 0; i < ids.length; i++){
    var key = ids[i];
    var btn = document.getElementById('srcTab_' + key);
    var pnl = document.getElementById('srcPanel_' + key);
    if (btn){
      btn.className = (key === id) ? 'tab-btn is-active' : 'tab-btn';
    }
    if (pnl){
      pnl.style.display = (key === id) ? 'block' : 'none';
    }
  }
  return false;
}
window.switchSourceTab = switchSourceTab;

/* ----- Chart.js 圖表初始化 ----- */
function initCharts(){
  if (typeof Chart === 'undefined') {
    // Tailwind JIT 與 Chart.js 皆以 <script> 非 defer 載入，理論上此處已就緒；
    // 若仍未定義代表腳本載入順序或路徑有誤。
    console.warn('[app.js] Chart.js 未載入，跳過圖表渲染');
    return;
  }

  // Stars chart（蝦皮／momo 採樣星等分布）
  var ctx1 = document.getElementById('chartStars');
  if (ctx1){
    new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: ['1 星','2 星','3 星','4 星','5 星'],
        datasets: [{
          label: '蝦皮 / momo 評論',
          data: [89, 29, 93, 70, 102],
          backgroundColor: ['#be123c','#c2410c','#b45309','#65a30d','#047857'],
          borderRadius: 8,
          barThickness: 40
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: '#f5f5f4' } },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // Top 10 chart（跨來源綜合痛點合併分數 — 水平長條）
  var ctx2 = document.getElementById('chartTop10');
  if (ctx2){
    var data = [
      { label: '價格太貴 / 智商稅',  score: 78 },
      { label: '亮度不足 / 太暗',    score: 66 },
      { label: '故障 / 燒壞 / 短壽', score: 62 },
      { label: '反光到螢幕',         score: 60 },
      { label: '充電 / 觸控失靈',    score: 54 },
      { label: '刺眼 / 眩光',        score: 48 },
      { label: '做工差 / 底座不穩',  score: 42 },
      { label: '客服差 / 退貨難',    score: 38 },
      { label: '瑕疵品 / 少件',      score: 36 },
      { label: '頻閃 / 閃爍',        score: 30 }
    ];
    new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: data.map(function(d){ return d.label; }),
        datasets: [{
          label: '合併痛點分數',
          data: data.map(function(d){ return d.score; }),
          backgroundColor: ['#be123c','#c2410c','#b45309','#a16207','#0369a1','#475569','#6d28d9','#be185d','#ea580c','#78716c'],
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: function(c){ return ' 分數：' + c.parsed.x; } } }
        },
        scales: {
          x: { beginAtZero: true, grid: { color: '#f5f5f4' } },
          y: { grid: { display: false }, ticks: { font: { size: 13 } } }
        }
      }
    });
  }
}

if (document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', initCharts);
} else {
  initCharts();
}
