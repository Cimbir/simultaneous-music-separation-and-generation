export const byId = (id) => document.getElementById(id);

export function setText(id, value) {
  const el = byId(id);
  if (el) el.textContent = value;
}

const screens = new Map();

export function registerScreens(names) {
  names.forEach((name) => screens.set(name, byId(`screen-${name}`)));
}

export function showScreen(name) {
  for (const el of screens.values()) el.classList.remove("active");
  screens.get(name).classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}
