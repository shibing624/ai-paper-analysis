// 按需加载 MathJax：只有页面里出现 .arithmatex（pymdownx.arithmatex generic 模式
// 会把所有 LaTeX 公式包成 <span class="arithmatex">\(...\)</span> 或 div 形式）才加载。
// 没有公式的页面不加载，省 ~700KB JS + 一次 CDN 往返。

(function () {
  'use strict';

  function load() {
    if (!document.querySelector('.arithmatex')) return;

    window.MathJax = {
      tex: {
        inlineMath: [['\\(', '\\)']],
        displayMath: [['\\[', '\\]']],
        processEscapes: true,
      },
      options: {
        ignoreHtmlClass: 'no-mathjax',
        processHtmlClass: 'arithmatex',
      },
      svg: { fontCache: 'global' },
    };

    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    s.async = true;
    document.head.appendChild(s);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    load();
  }
})();
