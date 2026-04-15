
(function(){
  function detectLang(){
    const saved = localStorage.getItem('study_lang');
    if (saved === 'fr' || saved === 'en') return saved;
    const nav = (navigator.language || 'en').toLowerCase();
    return nav.startsWith('fr') ? 'fr' : 'en';
  }
  function t(key){
    const lang = window.currentLang || detectLang();
    return (window.I18N?.[lang]?.[key]) || key;
  }
  function applyLang(lang){
    window.currentLang = lang;
    localStorage.setItem('study_lang', lang);
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (key) el.textContent = t(key);
    });
    document.querySelectorAll('.lang-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    if (window.onLanguageApplied) window.onLanguageApplied(lang);
  }
  window.detectLang = detectLang;
  window.t = t;
  window.applyLang = applyLang;
})();
