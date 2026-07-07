const PLAY_ICON_SRC = new URL("../assets/icons/play.svg", import.meta.url).href;
const PAUSE_ICON_SRC = new URL("../assets/icons/pause.svg", import.meta.url).href;

function setButtonIcon(button, src, label) {
  button.replaceChildren();

  const icon = document.createElement("img");
  icon.className = "button-icon";
  icon.src = src;
  icon.alt = "";
  icon.setAttribute("aria-hidden", "true");

  button.appendChild(icon);
  if (label) button.setAttribute("aria-label", label);
}

export function setPlayIcon(button, label) {
  setButtonIcon(button, PLAY_ICON_SRC, label);
}

export function setPauseIcon(button, label) {
  setButtonIcon(button, PAUSE_ICON_SRC, label);
}
