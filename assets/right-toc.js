// 在右侧固定一个"本页目录"，类似 Material 主题。
// 实现：扫描正文里所有带 id 的 h2/h3，构建一个 fixed 定位的 nav；
// 滚动时高亮当前可见的 section（scrollspy）。

(function () {
  'use strict';

  function build() {
    var article = document.querySelector('div[role="main"]');
    if (!article) return;

    // 排除首页之类的列表页，至少要有 3 个 h2 才显示
    var headings = article.querySelectorAll('h2[id], h3[id]');
    if (headings.length < 3) return;

    var nav = document.createElement('nav');
    nav.className = 'right-toc';

    var title = document.createElement('div');
    title.className = 'right-toc-title';
    title.textContent = '本页目录';
    nav.appendChild(title);

    var ul = document.createElement('ul');
    headings.forEach(function (h) {
      var li = document.createElement('li');
      li.className = 'right-toc-' + h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent.replace(/[¶#]/g, '').trim();
      a.dataset.targetId = h.id;
      li.appendChild(a);
      ul.appendChild(li);
    });
    nav.appendChild(ul);
    document.body.appendChild(nav);

    var links = nav.querySelectorAll('a');
    var headingArr = Array.prototype.slice.call(headings);

    function onScroll() {
      var scrollPos = window.scrollY + 140;
      var current = headingArr[0].id;
      for (var i = 0; i < headingArr.length; i++) {
        if (headingArr[i].offsetTop <= scrollPos) {
          current = headingArr[i].id;
        } else {
          break;
        }
      }
      links.forEach(function (l) {
        if (l.dataset.targetId === current) {
          l.classList.add('active');
          // 让当前项滚到 toc 可视区
          var rect = l.getBoundingClientRect();
          var navRect = nav.getBoundingClientRect();
          if (rect.bottom > navRect.bottom || rect.top < navRect.top) {
            l.scrollIntoView({ block: 'nearest' });
          }
        } else {
          l.classList.remove('active');
        }
      });
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();
