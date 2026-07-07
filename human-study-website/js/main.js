import { createStudyApi } from "./api.js";
import { byId, setText, registerScreens, showScreen } from "./ui.js";
import { createTrial } from "./trial.js";
import { setPauseIcon, setPlayIcon } from "./icons.js";

const CONFIG = window.STUDY_CONFIG || {};

const CALIBRATION_CLIPS = [
  {
    id: "gt-reference",
    badge: "Reference",
    title: "Good realism + good coherence",
    src: "audio/instructions/gt_reference.mp3",
    description:
      "A real ground-truth mix. The instruments sound believable and they fit together as one song.",
  },
  {
    id: "bad-realism-good-coherence",
    badge: "Anchor A",
    title: "Bad realism + good coherence",
    src: "audio/instructions/bad_realism_good_coherence.mp3",
    description:
      "The song structure still fits together, but the instrument sounds are damaged and synthetic.",
  },
  {
    id: "good-realism-bad-coherence",
    badge: "Anchor B",
    title: "Good realism + bad coherence",
    src: "audio/instructions/good_realism_bad_coherence.mp3",
    description:
      "The layers come from real ground-truth material, but they do not share one song, groove, or phrase.",
  },
];

const trialRefs = {
  players: byId("players"),
  musicalityList: byId("rank-musicality"),
  coherenceList: byId("rank-coherence"),
};

const study = {
  meta: null,
  api: null,
  session: null,
  participantId: null,
  sessionIndex: null,
  backendReady: false,
  maxPlays: 3,
  trialIndex: 0,
  practiceMode: false,
  trial: null,
  stopCalibrationAudio: null,
};

async function init() {
  registerScreens(["welcome", "instructions", "calibration", "trial", "practice-done", "done", "error"]);
  study.api = createStudyApi(CONFIG);
  applyBranding();

  try {
    study.meta = await loadSessionPlan();
  } catch (err) {
    console.error(err);
    return showError("Could not load study data. Make sure data/sessions.json exists (run build_sessions.py).");
  }

  study.maxPlays = study.meta.plays_per_clip || 3;
  setText("playsInfo", study.maxPlays);
  setText("realCount", study.meta.sessions[0]?.trials.length || 7);

  setupCalibrationExamples();
  wireButtons();
  showScreen("welcome");
}

function applyBranding() {
  setText("estMin", CONFIG.ESTIMATED_MINUTES ?? 10);
  if (CONFIG.STUDY_TITLE) {
    setText("brand", CONFIG.STUDY_TITLE);
    document.title = CONFIG.STUDY_TITLE;
  }
}

async function loadSessionPlan() {
  const res = await fetch("data/sessions.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`sessions.json ${res.status}`);
  return res.json();
}

function wireButtons() {
  byId("startBtn").addEventListener("click", onBegin);
  byId("toCalibrationBtn").addEventListener("click", () => showScreen("calibration"));
  byId("toPracticeBtn").addEventListener("click", () => {
    study.stopCalibrationAudio?.();
    runTrial(makePracticeClips(), { practice: true });
  });
  byId("toStudyBtn").addEventListener("click", () => {
    study.trialIndex = 0;
    runTrial(study.session.trials[0].clips, { practice: false });
  });
  byId("nextTrialBtn").addEventListener("click", onContinue);
  byId("retryBtn").addEventListener("click", () => location.reload());
}

function setupCalibrationExamples() {
  const container = byId("calibrationExamples");
  const played = new Set();
  let activeAudio = null;
  let activeUi = null;

  container.innerHTML = "";

  function stopActive() {
    if (!activeAudio) return;
    activeAudio.pause();
    activeAudio.currentTime = 0;
    if (activeUi) {
      setPlayIcon(activeUi.button, activeUi.button.dataset.playLabel);
      activeUi.progressFill.style.width = "0%";
    }
    activeAudio = null;
    activeUi = null;
  }

  function updateReadyState() {
    const ready = played.size === CALIBRATION_CLIPS.length;
    byId("toPracticeBtn").disabled = !ready;
    setText(
      "calibrationWarn",
      ready ? "You can start the practice round now." : "Play all three examples before practice.",
    );
  }

  for (const clip of CALIBRATION_CLIPS) {
    const ui = renderCalibrationExample(clip);
    const audio = new Audio(clip.src);
    audio.preload = "auto";

    audio.addEventListener("timeupdate", () => {
      if (audio.duration) {
        ui.progressFill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
      }
    });
    audio.addEventListener("ended", () => {
      setPlayIcon(ui.button, ui.button.dataset.playLabel);
      ui.progressFill.style.width = "0%";
      if (activeAudio === audio) {
        activeAudio = null;
        activeUi = null;
      }
    });

    ui.button.addEventListener("click", () => {
      if (!audio.paused) {
        stopActive();
        return;
      }

      stopActive();
      audio.currentTime = 0;
      audio.play().then(() => {
        activeAudio = audio;
        activeUi = ui;
        setPauseIcon(ui.button, ui.button.dataset.pauseLabel);
        ui.row.classList.add("played");
        ui.status.textContent = "Played";
        played.add(clip.id);
        updateReadyState();
      }).catch((err) => {
        console.error(err);
        setText("calibrationWarn", "Could not play this example. Check the audio files and try again.");
      });
    });

    container.appendChild(ui.row);
  }

  study.stopCalibrationAudio = stopActive;
  updateReadyState();
}

function renderCalibrationExample(clip) {
  const row = document.createElement("article");
  row.className = "calibration-example";
  row.innerHTML = `
    <div class="calibration-copy">
      <span class="badge">${clip.badge}</span>
      <h3>${clip.title}</h3>
      <p>${clip.description}</p>
    </div>
    <div class="calibration-player">
      <button class="calibration-play" type="button"></button>
      <div class="calibration-meter">
        <div class="wave"><i></i></div>
        <div class="calibration-status">Not played yet</div>
      </div>
    </div>`;

  const button = row.querySelector(".calibration-play");
  button.dataset.playLabel = `Play ${clip.title}`;
  button.dataset.pauseLabel = `Pause ${clip.title}`;
  setPlayIcon(button, button.dataset.playLabel);

  return {
    row,
    button,
    progressFill: row.querySelector(".wave > i"),
    status: row.querySelector(".calibration-status"),
  };
}

async function onBegin() {
  const button = byId("startBtn");
  button.disabled = true;
  button.textContent = "Loading…";
  await assignSession();
  showScreen("instructions");
}

async function assignSession() {
  if (CONFIG.USE_SUPABASE) {
    try {
      const { participantId, sessionIndex } = await study.api.claimSession(study.meta.num_sessions);
      study.participantId = participantId;
      study.sessionIndex = sessionIndex;
      study.backendReady = true;
    } catch (err) {
      console.error(err);
      study.backendReady = false;
      study.sessionIndex = randomSessionIndex();
    }
  } else {
    study.sessionIndex = randomSessionIndex();
  }

  study.session =
    study.meta.sessions.find((s) => s.session_index === study.sessionIndex) ||
    study.meta.sessions[0];
}

const randomSessionIndex = () => Math.floor(Math.random() * study.meta.num_sessions);

function makePracticeClips() {
  const clips = study.meta.practice.clips.map((clip) => ({ ...clip }));
  for (let i = clips.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [clips[i], clips[j]] = [clips[j], clips[i]];
  }
  return clips;
}

function runTrial(clips, { practice }) {
  study.practiceMode = practice;
  study.trial = createTrial(trialRefs, clips, {
    maxPlays: study.maxPlays,
    onReadyChange: setContinueEnabled,
  });

  setText("trialBadge", practice ? "Practice" : `Round ${study.trialIndex + 1}`);
  setText("trialTitle", "Listen to all five clips, then rank them");
  byId("nextTrialBtn").textContent = continueLabel(practice);

  setContinueEnabled(false);
  if (!practice) updateProgress();
  showScreen("trial");
}

function continueLabel(practice) {
  if (practice) return "Finish practice";
  const isLastTrial = study.trialIndex + 1 >= study.session.trials.length;
  return isLastTrial ? "Submit & finish" : "Continue";
}

function setContinueEnabled(ready) {
  byId("nextTrialBtn").disabled = !ready;
  setText("trialWarn", ready ? "" : "Please play every clip at least once before continuing.");
}

function updateProgress() {
  const total = study.session.trials.length;
  const done = study.trialIndex;
  byId("progress").hidden = false;
  byId("progressFill").style.width = `${(done / total) * 100}%`;
  setText("progressLabel", `Round ${Math.min(done + 1, total)} of ${total}`);
}

async function onContinue() {
  if (!study.trial.isReady()) return;
  study.trial.stopAudio();

  if (study.practiceMode) {
    showScreen("practice-done");
    return;
  }

  byId("nextTrialBtn").disabled = true;
  await saveCurrentTrial();

  study.trialIndex += 1;
  if (study.trialIndex >= study.session.trials.length) {
    await finishStudy();
  } else {
    runTrial(study.session.trials[study.trialIndex].clips, { practice: false });
  }
}

async function saveCurrentTrial() {
  if (!study.backendReady) return;
  const trialMeta = study.session.trials[study.trialIndex];
  const answers = study.trial.answers();
  try {
    await study.api.saveResponse({
      participant_id: study.participantId,
      trial_number: trialMeta.trial_number,
      is_catch: !!trialMeta.is_catch,
      clips: answers.clips,
      musicality: answers.musicality,
      coherence: answers.coherence,
      replays: answers.replays,
      trial_started_at: answers.startedAt,
    });
  } catch (err) {
    console.error(err);
  }
}

async function finishStudy() {
  if (study.backendReady) {
    try {
      await study.api.markParticipantFinished(study.participantId);
    } catch (err) {
      console.error(err);
    }
  }
  byId("progressFill").style.width = "100%";
  setText("progressLabel", "Done");
  showScreen("done");
}

function showError(message) {
  setText("errorMsg", message);
  showScreen("error");
}

document.addEventListener("DOMContentLoaded", init);
