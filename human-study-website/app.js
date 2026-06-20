(() => {
  "use strict";
  const CFG = window.STUDY_CONFIG || {};
  const LABELS = ["A", "B", "C", "D"];

  const state = {
    sessionsMeta: null,
    session: null,
    participantId: null,
    sessionIndex: null,
    trialIndex: 0,
    current: null,
    backendOk: false,
    localLog: [],
  };

  const $ = (id) => document.getElementById(id);
  const screens = {};
  ["welcome", "instructions", "trial", "done", "error"].forEach(
    (k) => (screens[k] = $("screen-" + k))
  );

  function show(name) {
    Object.values(screens).forEach((s) => s.classList.remove("active"));
    screens[name].classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function fail(msg) {
    $("errorMsg").textContent = msg || "Unexpected error.";
    show("error");
  }

  async function sb(path, opts = {}) {
    const res = await fetch(`${CFG.SUPABASE_URL}/rest/v1/${path}`, {
      ...opts,
      headers: {
        apikey: CFG.SUPABASE_ANON_KEY,
        Authorization: `Bearer ${CFG.SUPABASE_ANON_KEY}`,
        "Content-Type": "application/json",
        Prefer: "return=representation",
        ...(opts.headers || {}),
      },
    });
    if (!res.ok) throw new Error(`Supabase ${res.status}: ${await res.text()}`);
    return res.status === 204 ? null : res.json();
  }

  async function claimSession() {
    const rows = await sb("rpc/claim_session", {
      method: "POST",
      body: JSON.stringify({
        p_num_sessions: state.sessionsMeta.num_sessions,
        p_user_agent: navigator.userAgent.slice(0, 300),
      }),
    });
    const row = Array.isArray(rows) ? rows[0] : rows;
    state.participantId = row.participant_id;
    state.sessionIndex = row.session_index;
  }

  async function saveTrial(payload) {
    if (state.backendOk) {
      try {
        await sb("responses", { method: "POST", body: JSON.stringify(payload) });
        return;
      } catch (e) {
        console.error(e);
        state.backendOk = false;
      }
    }
    state.localLog.push(payload);
  }

  async function markFinished() {
    if (state.backendOk) {
      try {
        await sb(`participants?id=eq.${state.participantId}`, {
          method: "PATCH",
          body: JSON.stringify({ finished_at: new Date().toISOString() }),
        });
        return;
      } catch (e) {
        console.error(e);
        state.backendOk = false;
      }
    }
    downloadLocal();
  }

  function downloadLocal() {
    const blob = new Blob(
      [JSON.stringify({
        session_index: state.sessionIndex,
        finished_at: new Date().toISOString(),
        responses: state.localLog,
      }, null, 2)],
      { type: "application/json" }
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `study_session_${state.sessionIndex ?? "x"}_${Date.now()}.json`;
    a.click();
    $("doneNote").textContent =
      "A results file was downloaded — please send it to the researcher.";
  }

  function updateProgress() {
    const total = state.session.trials.length;
    const done = state.trialIndex;
    $("progress").hidden = false;
    $("progressFill").style.width = `${(done / total) * 100}%`;
    $("progressLabel").textContent = `Round ${Math.min(done + 1, total)} of ${total}`;
  }

  let activeAudio = null;

  function buildPlayers(clips) {
    const wrap = $("players");
    wrap.innerHTML = "";
    state.current.plays = {};
    state.current.audios = {};
    clips.forEach((clip, i) => {
      const label = LABELS[i];
      state.current.plays[clip.clip_id] = 0;
      const audio = new Audio(clip.src);
      audio.preload = "auto";
      state.current.audios[clip.clip_id] = audio;

      const row = document.createElement("div");
      row.className = "player";
      row.innerHTML = `
        <button class="play" aria-label="Play clip ${label}">▶</button>
        <div class="label">${label}</div>
        <div class="meta">
          <div class="wave"><i></i></div>
        </div>`;
      const btn = row.querySelector(".play");
      const fill = row.querySelector(".wave > i");

      audio.addEventListener("timeupdate", () => {
        if (audio.duration) fill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
      });
      audio.addEventListener("ended", () => {
        btn.textContent = "▶";
        fill.style.width = "0%";
      });

      btn.addEventListener("click", () => {
        if (activeAudio && activeAudio !== audio) {
          activeAudio.pause();
          activeAudio.currentTime = 0;
        }
        if (!audio.paused) {
          audio.pause();
          btn.textContent = "▶";
          return;
        }
        state.current.plays[clip.clip_id] += 1;
        audio.currentTime = 0;
        audio.play();
        activeAudio = audio;
        btn.textContent = "⏸";
        row.classList.add("played");
        validateTrial();
      });

      wrap.appendChild(row);
    });
  }

  function buildRanklist(elId, clips, orderKey) {
    const ul = $(elId);
    ul.innerHTML = "";
    state.current[orderKey].forEach((clipId) => {
      const idx = clips.findIndex((c) => c.clip_id === clipId);
      const li = document.createElement("li");
      li.className = "rankitem";
      li.draggable = true;
      li.dataset.clip = clipId;
      li.innerHTML = `
        <span class="rank-num"></span>
        <span class="rank-name">Clip ${LABELS[idx]}</span>
        <span class="arrows">
          <button class="up" aria-label="Move up">▲</button>
          <button class="down" aria-label="Move down">▼</button>
        </span>
        <span class="grip">⋮⋮</span>`;
      li.querySelector(".up").addEventListener("click", () => move(orderKey, clipId, -1, clips, elId));
      li.querySelector(".down").addEventListener("click", () => move(orderKey, clipId, 1, clips, elId));
      addDnd(li, orderKey, clips, elId);
      ul.appendChild(li);
    });
    renumber(ul);
  }

  function move(orderKey, clipId, dir, clips, elId) {
    const arr = state.current[orderKey];
    const i = arr.indexOf(clipId);
    const j = i + dir;
    if (j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    buildRanklist(elId, clips, orderKey);
  }

  let dragSrc = null;
  function addDnd(li, orderKey, clips, elId) {
    li.addEventListener("dragstart", () => {
      dragSrc = li;
      li.classList.add("dragging");
    });
    li.addEventListener("dragend", () => {
      li.classList.remove("dragging");
      dragSrc = null;
    });
    li.addEventListener("dragover", (e) => {
      e.preventDefault();
      li.classList.add("over");
    });
    li.addEventListener("dragleave", () => li.classList.remove("over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      li.classList.remove("over");
      if (!dragSrc || dragSrc === li) return;
      const arr = state.current[orderKey];
      const from = arr.indexOf(dragSrc.dataset.clip);
      const to = arr.indexOf(li.dataset.clip);
      arr.splice(to, 0, arr.splice(from, 1)[0]);
      buildRanklist(elId, clips, orderKey);
    });
  }

  function renumber(ul) {
    [...ul.children].forEach((li, i) => {
      li.querySelector(".rank-num").textContent = i + 1;
      li.querySelector(".up").disabled = i === 0;
      li.querySelector(".down").disabled = i === ul.children.length - 1;
    });
  }

  function allPlayed() {
    return Object.values(state.current.plays).every((n) => n >= 1);
  }

  function validateTrial() {
    const ok = allPlayed();
    $("nextTrialBtn").disabled = !ok;
    $("trialWarn").textContent = ok ? "" : "Please play every clip at least once before continuing.";
    return ok;
  }

  function renderTrial(trial) {
    const clips = trial.clips;
    state.current = {
      clips,
      orderM: clips.map((c) => c.clip_id),
      orderC: clips.map((c) => c.clip_id),
      startedAt: new Date().toISOString(),
    };
    $("trialBadge").textContent = `Round ${state.trialIndex + 1}`;
    $("trialTitle").textContent = "Listen to all four clips, then rank them";
    $("nextTrialBtn").textContent =
      state.trialIndex + 1 >= state.session.trials.length
        ? "Submit & finish"
        : "Continue";

    buildPlayers(clips);
    buildRanklist("rank-musicality", clips, "orderM");
    buildRanklist("rank-coherence", clips, "orderC");
    validateTrial();
    updateProgress();
    show("trial");
  }

  async function submitTrial() {
    if (!validateTrial()) return;
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }

    const trial = state.session.trials[state.trialIndex];
    const payload = {
      participant_id: state.participantId,
      trial_number: trial.trial_number,
      is_catch: !!trial.is_catch,
      clips: state.current.clips.map((c, i) => ({
        clip_id: c.clip_id, model: c.model, src: c.src, position: i,
      })),
      musicality: state.current.orderM,
      coherence: state.current.orderC,
      replays: state.current.plays,
      trial_started_at: state.current.startedAt,
    };

    $("nextTrialBtn").disabled = true;
    await saveTrial(payload);

    state.trialIndex += 1;
    if (state.trialIndex >= state.session.trials.length) {
      await markFinished();
      $("progressFill").style.width = "100%";
      $("progressLabel").textContent = "Done";
      show("done");
    } else {
      renderTrial(state.session.trials[state.trialIndex]);
    }
  }

  async function loadJSON(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }

  async function init() {
    $("estMin").textContent = CFG.ESTIMATED_MINUTES ?? 10;
    if (CFG.STUDY_TITLE) {
      $("brand").textContent = CFG.STUDY_TITLE;
      document.title = CFG.STUDY_TITLE;
    }

    try {
      const meta = await loadJSON("data/sessions.json");
      state.sessionsMeta = meta;
      $("realCount").textContent = meta.sessions[0]?.trials.length || 7;
    } catch (e) {
      console.error(e);
      return fail("Could not load study data. Make sure data/sessions.json exists (run build_sessions.py).");
    }

    $("startBtn").addEventListener("click", onStart);
    $("toStudyBtn").addEventListener("click", () => {
      state.trialIndex = 0;
      renderTrial(state.session.trials[0]);
    });
    $("nextTrialBtn").addEventListener("click", submitTrial);
    $("retryBtn").addEventListener("click", () => location.reload());

    show("welcome");
  }

  async function onStart() {
    $("startBtn").disabled = true;
    $("startBtn").textContent = "Loading…";

    if (CFG.USE_SUPABASE) {
      try {
        await claimSession();
        state.backendOk = true;
      } catch (e) {
        console.error(e);
        state.backendOk = false;
        state.sessionIndex = Math.floor(Math.random() * state.sessionsMeta.num_sessions);
      }
    } else {
      state.sessionIndex = Math.floor(Math.random() * state.sessionsMeta.num_sessions);
    }

    state.session =
      state.sessionsMeta.sessions.find((s) => s.session_index === state.sessionIndex) ||
      state.sessionsMeta.sessions[0];

    show("instructions");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
