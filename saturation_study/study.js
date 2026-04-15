
const setup = JSON.parse(sessionStorage.getItem("study_setup") || "null");
if (!setup) location.href = "index.html";

const cfg = window.STUDY_CONFIG;
const API_BASE = `${cfg.supabaseUrl}/functions/v1`;

let participantId = null;
let anonymousCode = null;
let sessionId = null;
let trialIndex = 0;
let startedAt = 0;
let currentTrial = null;

function rgbCss(rgb) {
  return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
}
function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}
function hslToRgb(h, s, l) {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    hue2rgb(p, q, h + 1 / 3),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1 / 3),
  ];
}
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
async function post(path, body) {
  const res = await fetch(`${API_BASE}/${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey": cfg.publishableKey,
    },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
async function initParticipantAndSession() {
  const participant = await post("create-participant", {
    language: navigator.language,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    user_agent: navigator.userAgent,
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    pixel_ratio: window.devicePixelRatio || 1,
    platform: navigator.platform,
    study_version: setup.study_version,
  });
  participantId = participant.participant_id;
  anonymousCode = participant.anonymous_code;

  const session = await post("create-session", {
    participant_id: participantId,
    patch_count: setup.patch_count,
    series_count: setup.series_count,
    consent_given: setup.consent_given,
    test_instruction: setup.test_instruction,
    protocol_name: setup.protocol_name,
    session_seed: setup.session_seed,
  });
  sessionId = session.session_id;
}
function buildTrial() {
  const family = cfg.families[Math.floor(Math.random() * cfg.families.length)];
  const sats = cfg.saturationGrid[setup.patch_count];
  const ordered = shuffle(sats.map((sat, idx) => ({ idx, sat })));

  return {
    family_name: window.currentLang === 'fr' ? family.name_fr : family.name_en,
    hue: family.h,
    lightness: family.l,
    reference_saturation: cfg.referenceSaturation,
    candidate_saturations: sats,
    candidate_order: ordered.map((x) => x.idx),
    displayed: ordered.map((x) => ({
      sat: x.sat,
      rgb: hslToRgb(family.h, x.sat, family.l),
    })),
  };
}
function renderTrial() {
  if (trialIndex >= setup.series_count) {
    finishStudy();
    return;
  }

  currentTrial = buildTrial();
  document.getElementById("instruction").textContent = t('instruction');
  document.getElementById("progress").textContent = `${t('trial')} ${trialIndex + 1} / ${setup.series_count}`;
  document.getElementById("anonCode").textContent = anonymousCode || "…";
  document.getElementById("counts").textContent = `${setup.patch_count} ${t('patches').toLowerCase()} · ${setup.series_count} ${t('series').toLowerCase()}`;

  const ref = hslToRgb(
    currentTrial.hue,
    currentTrial.reference_saturation,
    currentTrial.lightness,
  );
  document.getElementById("reference").style.background = rgbCss(ref);
  document.getElementById("referenceMeta").textContent =
    `${currentTrial.family_name} — ${t('fixedHL')}`;

  const box = document.getElementById("swatches");
  box.innerHTML = "";
  currentTrial.displayed.forEach((patch, displayedIndex) => {
    const el = document.createElement("div");
    el.className = "swatch";
    el.style.background = rgbCss(patch.rgb);
    el.addEventListener("click", () => submitChoice(displayedIndex));
    box.appendChild(el);
  });

  startedAt = performance.now();
}
async function submitChoice(displayedIndex) {
  document.getElementById("error").textContent = "";
  const rt = Math.round(performance.now() - startedAt);
  const selectedSat = currentTrial.displayed[displayedIndex].sat;

  try {
    await post("create-trial", {
      session_id: sessionId,
      trial_index: trialIndex,
      family_name: currentTrial.family_name,
      hue: currentTrial.hue,
      lightness: currentTrial.lightness,
      reference_saturation: currentTrial.reference_saturation,
      candidate_saturations: currentTrial.candidate_saturations,
      candidate_order: currentTrial.candidate_order,
      selected_index: displayedIndex,
      selected_saturation: selectedSat,
      reaction_time_ms: rt,
      pointer_type: "mouse",
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
    });
    trialIndex += 1;
    renderTrial();
  } catch (err) {
    document.getElementById("error").textContent = `${t('saveError')} : ${err.message}`;
  }
}
async function finishStudy() {
  document.getElementById("progress").textContent = t('finalizing');
  try {
    const summary = await post("complete-session", { session_id: sessionId });
    sessionStorage.setItem(
      "study_summary",
      JSON.stringify({
        session_id: sessionId,
        participant_id: participantId,
        anonymous_code: anonymousCode,
        summary,
      }),
    );
    location.href = "thankyou.html";
  } catch (err) {
    document.getElementById("error").textContent = `${t('completeError')} : ${err.message}`;
  }
}
window.onLanguageApplied = function(){
  if (participantId) renderTrial();
  document.querySelector('[data-i18n="testTitle"]').textContent = t('testTitle');
  document.querySelector('[data-i18n="siteTagline"]').textContent = t('siteTagline');
  document.querySelector('[data-i18n="referenceTitle"]').textContent = t('referenceTitle');
  document.getElementById("instruction").textContent = t('instruction');
};
(async function boot() {
  try {
    await initParticipantAndSession();
    renderTrial();
  } catch (err) {
    document.getElementById("error").textContent = `${t('initError')} : ${err.message}`;
  }
})();
