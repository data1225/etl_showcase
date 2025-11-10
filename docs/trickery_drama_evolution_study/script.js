  // Define global variables
  window.timeData = {
    lastYear: new Date().getFullYear() - 1,
    twoYearsAgo: new Date().getFullYear() - 2,
    threeYearsAgo: new Date().getFullYear() - 3,
    todayDate: new Date().toISOString().split('T')[0],
  };
  
  document.addEventListener('DOMContentLoaded', () => {
    
    // --- Inject year --- //
    // Inject year into text spans
    document.querySelectorAll('[data-year-text]').forEach(el => {
      const key = el.getAttribute('data-year-text');
      if (window.timeData[key] !== undefined) {
        el.textContent = window.timeData[key];
      }
    });
  
    // Replace {year} in anchor hrefs
    document.querySelectorAll('[data-year-href]').forEach(el => {
      const key = el.getAttribute('data-year-href');
      const href = el.getAttribute('href');
      if (window.timeData[key] !== undefined && href.includes('{year}')) {
        // el.setAttribute('href', href.replace('{year}', window.timeData[key]));
        el.setAttribute('href', href.replace(/{year}/g, window.timeData[key]));
      }
    });

    // Replace {date} in anchor hrefs
    document.querySelectorAll('[data-date-href]').forEach(el => {
      const key = el.getAttribute('data-date-href');
      const href = el.getAttribute('href');
      if (window.timeData[key] !== undefined && href.includes('{date}')) {
        el.setAttribute('href', href.replace(/{date}/g, window.timeData[key]));
      }
    });    

    // Html example
    // <a data-year-href="lastYear" href="https://example.com/analysis/{year}" target="_blank"><span data-year-text="lastYear"></span> link</a>
    // <a data-year-href="twoYearsAgo" href="https://example.com/docs/{year}" target="_blank"><span data-year-text="twoYearsAgo"></span> link</a>
    // <a data-year-href="threeYearsAgo" href="https://example.com/docs/{year}" target="_blank"><span data-year-text="threeYearsAgo"></span> link</a>
    // <a data-date-href="todayDate" href="https://example.com/docs/{date}" target="_blank"><span data-date-text="todayDate"></span> link</a>
    // --- Inject year --- //

    // Smooth scrolling for TOC
    document.querySelectorAll('#toc a').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const id = link.getAttribute('href').substring(1);
        const target = document.getElementById(id);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
    // Smooth scrolling for other menu
    document.querySelectorAll('.menu a').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const id = link.getAttribute('href').substring(1);
        const target = document.getElementById(id);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  });
  
  // Other custom functions
  function scrollToToc() {
    const toc = document.getElementById("toc");
    if (toc) {
      toc.scrollIntoView({ behavior: "smooth" });
    } else {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }  