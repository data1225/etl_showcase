
// 滾動時根據可見區段高亮對應錨點
const links = document.querySelectorAll('.nav-floating a');
const sections = Array.from(links).map(l => document.getElementById(l.dataset.target));

function updateActive() {
  const scrollPos = window.scrollY + window.innerHeight / 3;
  let activeIdx = 0;
  sections.forEach((sec, i) => {
    if (sec && sec.offsetTop <= scrollPos) activeIdx = i;
  });
  links.forEach((l, i) => l.classList.toggle('active', i === activeIdx));
}

window.addEventListener('scroll', updateActive, { passive: true });
window.addEventListener('load', () => {
  // 支援透過網址 hash 直接跳轉
  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  }
  updateActive();
});
