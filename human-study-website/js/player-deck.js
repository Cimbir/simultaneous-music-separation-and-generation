const ICON = { play: "▶", pause: "⏸" };

export function createPlayerDeck(container, clips, { maxPlays, labelFor, onChange }) {
  container.innerHTML = "";
  const playCounts = new Map(clips.map((clip) => [clip.clip_id, 0]));
  let activeAudio = null;

  function stopActive() {
    if (!activeAudio) return;
    activeAudio.pause();
    activeAudio.currentTime = 0;
    activeAudio = null;
  }

  for (const clip of clips) {
    const ui = renderRow(labelFor(clip), maxPlays);
    const audio = new Audio(clip.src);
    audio.preload = "auto";

    audio.addEventListener("timeupdate", () => {
      if (audio.duration) {
        ui.progressFill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
      }
    });
    audio.addEventListener("ended", () => {
      ui.button.textContent = ICON.play;
      ui.progressFill.style.width = "0%";
    });

    ui.button.addEventListener("click", () => onPlayToggle(clip, audio, ui));
    container.appendChild(ui.row);
  }

  function onPlayToggle(clip, audio, ui) {
    if (!audio.paused) {
      audio.pause();
      ui.button.textContent = ICON.play;
      return;
    }
    const used = playCounts.get(clip.clip_id);
    if (used >= maxPlays) return;

    stopActive();
    playCounts.set(clip.clip_id, used + 1);
    audio.currentTime = 0;
    audio.play();
    activeAudio = audio;
    ui.button.textContent = ICON.pause;
    ui.row.classList.add("played");

    const left = maxPlays - playCounts.get(clip.clip_id);
    ui.playsLabel.textContent = left > 0 ? `Plays left: ${left}` : "No plays left";
    if (left <= 0) ui.button.disabled = true;
    onChange();
  }

  return {
    replayCounts: () => Object.fromEntries(playCounts),
    everyClipPlayed: () => [...playCounts.values()].every((n) => n >= 1),
    stop: stopActive,
  };
}

function renderRow(label, maxPlays) {
  const row = document.createElement("div");
  row.className = "player";
  row.innerHTML = `
    <button class="play" aria-label="Play clip ${label}">${ICON.play}</button>
    <div class="label">${label}</div>
    <div class="meta">
      <div class="wave"><i></i></div>
      <div class="plays">Plays left: ${maxPlays}</div>
    </div>`;
  return {
    row,
    button: row.querySelector(".play"),
    progressFill: row.querySelector(".wave > i"),
    playsLabel: row.querySelector(".plays"),
  };
}
