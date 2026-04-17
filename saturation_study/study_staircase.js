
const setup = JSON.parse(sessionStorage.getItem("study_setup") || "null");
if (!setup) location.href = "index.html";

const cfg = window.STUDY_CONFIG;
const sc = cfg.staircase;
const API_BASE = `${cfg.supabaseUrl}/functions/v1`;

let participantId = null;
let anonymousCode = null;
let sessionId = null;
let familyIndex = 0;
let globalTrialIndex = 0;
let currentFamily = null;
let state = null;
let startedAt = 0;

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
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

async function post(path, body) {
  const res = await fetch(`${API_BASE}/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "apikey": cfg.publishableKey },
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

function startFamily() {
  if (familyIndex >= cfg.families.length) { finishStudy(); return; }
  currentFamily = cfg.families[familyIndex];
  state = {
    level: sc.initialLevel,
    step: sc.initialStep,
    lastDirection: 0,
    reversals: [],
    trials: 0,
  };
  renderProgress();
  renderStimulus();
}

function renderProgress() {
  const box = document.getElementById("familyProgress");
  box.innerHTML = "";
  cfg.families.forEach((_, i) => {
    const d = document.createElement("span");
    d.className = "dot" + (i < familyIndex ? " done" : (i === familyIndex ? " current" : ""));
    box.appendChild(d);
  });
  const famName = window.currentLang === 'fr' ? currentFamily.name_fr : currentFamily.name_en;
  document.getElementById("familyBadge").textContent = `${t('family')} : ${famName}`;
  document.getElementById("anonCode").textContent = anonymousCode || "…";
  document.getElementById("instruction").textContent = t('staircaseInstruction');
}

function renderStimulus() {
  const rgb = hslToRgb(currentFamily.h, state.level, currentFamily.l);
  document.getElementById("stimulus").style.background = rgbCss(rgb);
  document.getElementById("progress").textContent =
    `${t('trial')} ${state.trials + 1}/${sc.maxTrialsPerFamily} — reversals ${state.reversals.length}/${sc.maxReversals} — sat=${state.level.toFixed(3)}`;
  console.log(`[staircase] family=${familyIndex} trial=${state.trials} sat=${state.level.toFixed(3)} step=${state.step.toFixed(3)} reversals=${state.reversals.length}`);
  startedAt = performance.now();
}

async function respond(kind) {
  document.getElementById("error").textContent = "";
  const rt = Math.round(performance.now() - startedAt);
  const displayedSat = state.level;

  const direction = (kind === "too_faded") ? +1 : -1;
  if (state.lastDirection !== 0 && direction !== state.lastDirection) {
    state.reversals.push(displayedSat);
    state.step *= sc.stepFactor;
  }
  state.lastDirection = direction;
  state.trials += 1;

  try {
    await post("create-trial", {
      session_id: sessionId,
      trial_index: globalTrialIndex,
      family_name: window.currentLang === 'fr' ? currentFamily.name_fr : currentFamily.name_en,
      hue: currentFamily.hue ?? currentFamily.h,
      lightness: currentFamily.l,
      reference_saturation: cfg.referenceSaturation,
      candidate_saturations: [displayedSat],
      candidate_order: [0],
      selected_index: 0,
      selected_saturation: displayedSat,
      reaction_time_ms: rt,
      pointer_type: `mouse:${kind}`,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
    });
    globalTrialIndex += 1;
  } catch (err) {
    document.getElementById("error").textContent = `${t('saveError')} : ${err.message}`;
    return;
  }

  const done = state.reversals.length >= sc.maxReversals || state.trials >= sc.maxTrialsPerFamily;
  if (done) {
    const n = Math.min(2, state.reversals.length);
    const threshold = n > 0
      ? state.reversals.slice(-n).reduce((a, b) => a + b, 0) / n
      : state.level;

    try {
      await post("create-trial", {
        session_id: sessionId,
        trial_index: globalTrialIndex,
        family_name: window.currentLang === 'fr' ? currentFamily.name_fr : currentFamily.name_en,
        hue: currentFamily.h,
        lightness: currentFamily.l,
        reference_saturation: cfg.referenceSaturation,
        candidate_saturations: state.reversals.length ? state.reversals : [state.level],
        candidate_order: (state.reversals.length ? state.reversals : [state.level]).map((_, i) => i),
        selected_index: 0,
        selected_saturation: threshold,
        reaction_time_ms: 0,
        pointer_type: "summary:threshold",
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
      });
      globalTrialIndex += 1;
    } catch (err) {
      document.getElementById("error").textContent = `${t('saveError')} : ${err.message}`;
    }

    familyIndex += 1;
    startFamily();
    return;
  }

  state.level = clamp(state.level + direction * state.step, sc.minSat, sc.maxSat);
  renderStimulus();
}

async function finishStudy() {
  document.getElementById("progress").textContent = t('finalizing');
  document.getElementById("stimulus").style.background = "transparent";
  document.getElementById("btnTooFaded").disabled = true;
  document.getElementById("btnTooVivid").disabled = true;
  try {
    const summary = await post("complete-session", { session_id: sessionId });
    sessionStorage.setItem("study_summary", JSON.stringify({
      session_id: sessionId,
      participant_id: participantId,
      anonymous_code: anonymousCode,
      summary,
    }));
    location.href = "thankyou.html";
  } catch (err) {
    document.getElementById("error").textContent = `${t('completeError')} : ${err.message}`;
  }
}

document.getElementById("btnTooFaded").addEventListener("click", () => respond("too_faded"));
document.getElementById("btnTooVivid").addEventListener("click", () => respond("too_vivid"));

window.onLanguageApplied = function(){
  document.querySelector('[data-i18n="siteTagline"]').textContent = t('siteTagline');
  if (currentFamily) renderProgress();
};

(async function boot() {
  try {
    await initParticipantAndSession();
    startFamily();
  } catch (err) {
    document.getElementById("error").textContent = `${t('initError')} : ${err.message}`;
  }
})();
